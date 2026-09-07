"""
build_save_graph.py
Construction et sauvegarde du graphe pour le dataset.
À exécuter une seule fois.
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pickle
import argparse
import yaml
from dataclasses import asdict
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports depuis les dossiers du projet
from models.encoders import FrozenAcousticBackbone
from graph.build_graph import GraphBuildConfig, build_heterogeneous_graph
from graph.similarity import (
    LinguisticRepresentation,
    compute_labse_embeddings,
    compute_afriberta_embeddings
)
from data.datasets import AudioTextDataset, Collator


# ============================================================================
# FONCTIONS DE CONVERSION
# ============================================================================

def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def to_int(value):
    if isinstance(value, (int, float)):
        return int(value)
    elif isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return value
    return value


def to_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return bool(value)
    elif isinstance(value, str):
        return value.lower() in ['true', '1', 'yes', 'on']
    return value


# ============================================================================
# GRAPH BUILDER
# ============================================================================

class GraphBuilder:
    def __init__(
        self,
        dataset: AudioTextDataset = None,
        output_dir: str = '',
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        backbone_name: str = "facebook/wav2vec2-xls-r-300m"
    ):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.backbone_name = backbone_name
        
        self.backbone = FrozenAcousticBackbone(backbone_name).to(device)
        self.backbone.eval()
        
        self.transcription_to_id = {}
        self.id_to_transcription = []
        self._build_transcription_mapping()
        
        print(f"✅ Dataset chargé: {len(self.dataset)} exemples")
        print(f"✅ {len(self.transcription_to_id)} transcriptions uniques")
        print(f"✅ Backbone gelé: {backbone_name}")
        print(f"✅ Device: {device}")
    
    def _build_transcription_mapping(self):
        all_transcriptions = self.dataset.get_all_transcriptions()
        unique_transcriptions = sorted(set(all_transcriptions))
        
        self.transcription_to_id = {
            trans: idx for idx, trans in enumerate(unique_transcriptions)
        }
        self.id_to_transcription = unique_transcriptions
        
        self.transcription_ids = torch.tensor(
            [self.transcription_to_id[trans] for trans in all_transcriptions],
            dtype=torch.long
        )
    
    def _compute_max_length(self) -> int:
        print("   Calcul de la longueur max du dataset...")
        max_len = 0
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            max_len = max(max_len, len(example.waveform))
        print(f"   Longueur max: {max_len} samples ({max_len/16000:.2f}s)")
        return max_len
    
    
    def extract_audio_features_from_path(self, audio_path: str) -> torch.Tensor:
        """
        Extrait les features acoustiques d'un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio
        
        Returns:
            torch.Tensor: Features acoustiques de forme (T, D) ou (D,) selon le pooling
        """
        import torchaudio
        import torch
        import librosa
        from pathlib import Path
        
        try:
            # 1. Charger le fichier audio
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
            
            # Essayer avec torchaudio d'abord (plus rapide)
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                # Si mono, prendre le premier canal
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.squeeze(0)  # (T,)
            except Exception as e:
                # Fallback sur librosa si torchaudio échoue
                print(f"⚠️ torchaudio a échoué, utilisation de librosa pour {audio_path}: {e}")
                waveform_np, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                waveform = torch.from_numpy(waveform_np).float()
            
            # 2. Resampler si nécessaire (à la fréquence du backbone)
            target_sample_rate = getattr(self, 'sample_rate', 16000)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
            
            # 3. Normaliser si configuré
            if getattr(self, 'normalize_audio', True):
                waveform = waveform / (waveform.abs().max() + 1e-8)
            
            # 4. S'assurer que la longueur est compatible
            # Si trop court, padder; si trop long, tronquer ou segmenter
            max_length = getattr(self, 'max_audio_length', None)
            if max_length is not None:
                if waveform.shape[0] < max_length:
                    # Padding
                    padding = max_length - waveform.shape[0]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                elif waveform.shape[0] > max_length:
                    # Tronquer ou prendre une fenêtre
                    waveform = waveform[:max_length]
            
            # 5. Ajouter la dimension batch pour le backbone
            waveform_batch = waveform.unsqueeze(0).to(self.device)  # (1, T)
            
            # 6. Extraire les features avec le backbone gelé
            with torch.no_grad():
                if hasattr(self, 'backbone') and self.backbone is not None:
                    # Utiliser le backbone existant
                    frame_features = self.backbone(waveform_batch, None)  # (1, T', D)
                else:
                    # Fallback: utiliser le backbone du builder
                    frame_features = self.builder.extract_frame_features(waveform_batch)
            
            # 7. Pooling temporel si nécessaire
            #pooling_method = getattr(self, 'pooling_method', 'mean')
            
            #if frame_features.dim() == 3:
                # (1, T, D) -> (T, D)
            #    frame_features = frame_features.squeeze(0)
                
            #    if pooling_method == 'mean':
            #        audio_features = frame_features.mean(dim=0)  # (D,)
            #    elif pooling_method == 'max':
            #        audio_features = frame_features.max(dim=0)[0]  # (D,)
            #   elif pooling_method == 'first':
            #        audio_features = frame_features[0, :]  # (D,)
            #    elif pooling_method == 'last':
            #        audio_features = frame_features[-1, :]  # (D,)
            #    else:
            #        raise ValueError(f"Méthode de pooling inconnue: {pooling_method}")
            #else:
                # Déjà poolé ou pas de dimension temporelle
            audio_features = frame_features.squeeze(0)
            
            return audio_features.cpu()
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'extraction des features de {audio_path}: {e}")
    
    def extract_frame_features(self, batch_size: int = 16, num_workers: int = 4):
        print("🔄 Extraction des features trames...")
        
        max_length = self._compute_max_length()
        
        collator = Collator(
            return_attention_mask=False,
            max_length=max_length
        )
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
        
        all_frame_features = []
        all_transcriptions = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extraction backbone")):
            waveforms = batch['waveforms'].to(self.device)
            
            if batch_idx == 0:
                print(f"\n[DEBUG] waveforms.shape: {waveforms.shape}")
            
            with torch.no_grad():
                frame_features = self.backbone(waveforms, None)
            
            if batch_idx == 0:
                print(f"[DEBUG] frame_features.shape: {frame_features.shape}")
            
            all_frame_features.append(frame_features.cpu())
            all_transcriptions.extend(batch['transcriptions'])
        
        frame_features = torch.cat(all_frame_features, dim=0)
        
        print(f"\n✅ Features trames: {frame_features.shape}")
        return frame_features, all_transcriptions
    
    def extract_linguistic_features(self, representation_type: LinguisticRepresentation = LinguisticRepresentation.LABSE):
        unique_transcriptions = self.id_to_transcription
        
        print(f"\n📝 Extraction features linguistiques ({representation_type.value})...")
        print(f"   {len(unique_transcriptions)} transcriptions uniques")
        
        if representation_type == LinguisticRepresentation.LABSE:
            features = compute_labse_embeddings(unique_transcriptions)

        elif representation_type == LinguisticRepresentation.AFRIBERTA:
            features = compute_afriberta_embeddings(unique_transcriptions)

        else:
            raise ValueError(f"Unknown representation: {representation_type}")
        
        print(f"✅ Features linguistiques: {features.shape}")
        return features
    
    def build_graph_from_frame_features(
        self,
        frame_features: torch.Tensor,
        linguistic_features: torch.Tensor,
        config: GraphBuildConfig,
    ):
        print(f"\n🏗️ Construction du graphe ...")
        
        #if pooling_method == "mean":
        #    audio_features = frame_features.mean(dim=1)
        #elif pooling_method == "max":
        #    audio_features = frame_features.max(dim=1)[0]
        #elif pooling_method == "first":
        #    audio_features = frame_features[:, 0, :]
        #else:
        #    raise ValueError(f"Méthode de pooling inconnue: {pooling_method}")
        
        print(f"   Features audio poolées: {frame_features.shape}")
        
        graph = build_heterogeneous_graph(
            audio_features=frame_features,
            linguistic_features=linguistic_features,
            audio_transcription_ids=self.transcription_ids,
            config=config
        )
        
        print(f"✅ Graphe construit:")
        print(f"   - Nœuds audio: {graph['audio'].x.shape[0]}")
        print(f"   - Nœuds linguistiques: {graph['word'].x.shape[0]}")
        print(f"   - Arêtes audio-audio: {graph['audio', 'similar_to', 'audio'].edge_index.shape[1]}")
        print(f"   - Arêtes linguistiques: {graph['word', 'similar_to', 'word'].edge_index.shape[1]}")
        print(f"   - Arêtes croisées: {graph['audio', 'transcribed_as', 'word'].edge_index.shape[1]}")
        
        return graph
    
    def build_and_save_all(
        self,
        config: GraphBuildConfig,
        representation_type: LinguisticRepresentation = LinguisticRepresentation.LABSE,
        batch_size: int = 16,
        num_workers: int = 4,
        pooling_method: str = "mean",
        save_features: bool = True,
        force_rebuild: bool = False
    ):
        features_dir = self.output_dir / 'features'
        model = self.backbone_name.split('/')[1]
        
        # FIX (cf. discussion, point 4) : le nom du fichier de cache du
        # graphe encode maintenant les hyperparamètres qui déterminent sa
        # construction (seuil, MST, méthode de backbone, pooling utilisé
        # pour les arêtes). AVANT, seuls `model` et `representation_type`
        # étaient encodés -- relancer avec un seuil différent sans
        # `force_rebuild=True` réutilisait SILENCIEUSEMENT l'ancien
        # graphe, sans avertissement, ce qui aurait invalidé toute
        # exploration de cet hyperparamètre en validation croisée.
        #mst_tag = f"mst{config.backbone_method}" if config.use_mst else "nomst"
        #graph_variant_tag = f"thr{config.similarity_threshold}_{mst_tag}"
        
        # Définir les chemins des fichiers
        frame_features_path = features_dir / f'frame_features_{model}.pt'
        linguistic_features_path = features_dir / f'linguistic_features_{representation_type}.pt'
        transcriptions_path = self.output_dir / 'transcriptions.txt'
        transcription_mapping_path = self.output_dir / 'transcription_mapping.pkl'
        transcription_ids_path = self.output_dir / 'transcription_ids.pt'
        graph_path = self.output_dir / 'graph' / f'hetero_graph_{model}_{representation_type}.pt'
        metadata_path = self.output_dir / f'metadata_{model}_{representation_type}.pkl'
        
        # Initialiser toutes les variables
        frame_features = None
        linguistic_features = None
        transcriptions = None
        graph = None
        metadata = None  # ← IMPORTANT: Initialiser metadata
        
        # 1. Vérifier et charger les frame features
        if frame_features_path.exists() and not force_rebuild:
            print(f"📂 Chargement des frame_features depuis {frame_features_path}")
            frame_features = torch.load(frame_features_path)
        else:
            print(f"🔨 Extraction des frame_features...")
            frame_features, transcriptions = self.extract_frame_features(
                batch_size=batch_size,
                num_workers=num_workers
            )
            if save_features:
                features_dir.mkdir(parents=True, exist_ok=True)
                torch.save(frame_features, frame_features_path)
                print(f"💾 frame_features sauvegardées")
        
        # 2. Vérifier et charger les linguistic features
        if linguistic_features_path.exists() and not force_rebuild:
            print(f"📂 Chargement des linguistic_features depuis {linguistic_features_path}")
            linguistic_features = torch.load(linguistic_features_path)
        else:
            print(f"🔨 Extraction des linguistic_features...")
            linguistic_features = self.extract_linguistic_features(
                representation_type=representation_type
            )
            if save_features:
                features_dir.mkdir(parents=True, exist_ok=True)
                torch.save(linguistic_features, linguistic_features_path)
                print(f"💾 linguistic_features sauvegardées")
        
        # 3. Vérifier et charger les transcriptions
        if transcriptions_path.exists() and not force_rebuild:
            print(f"📂 Chargement des transcriptions depuis {transcriptions_path}")
            with open(transcriptions_path, 'r', encoding='utf-8') as f:
                transcriptions = [line.strip() for line in f.readlines()]
        else:
            print(f"🔨 Sauvegarde des transcriptions...")
            if transcriptions is None:
                _, transcriptions = self.extract_frame_features(
                    batch_size=batch_size,
                    num_workers=num_workers
                )
            with open(transcriptions_path, 'w', encoding='utf-8') as f:
                for trans in transcriptions:
                    f.write(trans + '\n')
            print(f"💾 Transcriptions sauvegardées")
        
        # 4. Vérifier et charger le mapping
        if transcription_mapping_path.exists() and not force_rebuild:
            print(f"📂 Chargement du mapping depuis {transcription_mapping_path}")
            with open(transcription_mapping_path, 'rb') as f:
                mapping = pickle.load(f)
                self.transcription_to_id = mapping['transcription_to_id']
                self.id_to_transcription = mapping['id_to_transcription']
        else:
            print(f"🔨 Sauvegarde du mapping...")
            with open(transcription_mapping_path, 'wb') as f:
                pickle.dump({
                    'transcription_to_id': self.transcription_to_id,
                    'id_to_transcription': self.id_to_transcription
                }, f)
            print(f"💾 Mapping sauvegardé")
        
        # 5. Vérifier et charger les transcription_ids
        if transcription_ids_path.exists() and not force_rebuild:
            print(f"📂 Chargement des transcription_ids depuis {transcription_ids_path}")
            self.transcription_ids = torch.load(transcription_ids_path)
        else:
            print(f"🔨 Sauvegarde des transcription_ids...")
            torch.save(self.transcription_ids, transcription_ids_path)
            print(f"💾 transcription_ids sauvegardés")
        
        # 6. Vérifier et charger le graphe
        if graph_path.exists() and not force_rebuild:
            print(f"📂 Chargement du graphe depuis {graph_path}")
            graph = torch.load(graph_path)
        else:
            print(f"🔨 Construction du graphe...")
            if frame_features is None:
                frame_features = torch.load(frame_features_path)
            if linguistic_features is None:
                linguistic_features = torch.load(linguistic_features_path)
            
            graph = self.build_graph_from_frame_features(
                frame_features=frame_features,
                linguistic_features=linguistic_features,
                config=config
            )
            (self.output_dir / 'graph').mkdir(parents=True, exist_ok=True)
            torch.save(graph, graph_path)
            print(f"💾 Graphe sauvegardé")
        
        # 7. Vérifier et charger les métadonnées
        if metadata_path.exists() and not force_rebuild:
            print(f"📂 Chargement des métadonnées depuis {metadata_path}")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            print(f"🔨 Création des métadonnées...")
            if graph is None:
                graph = torch.load(graph_path)
            if frame_features is None:
                frame_features = torch.load(frame_features_path)
            if linguistic_features is None:
                linguistic_features = torch.load(linguistic_features_path)
            
            metadata = {
                'config': asdict(config),
                'num_audio_nodes': graph['audio'].x.shape[0],
                'num_word_nodes': graph['word'].x.shape[0],
                'num_audio_audio_edges': graph['audio', 'similar_to', 'audio'].edge_index.shape[1],
                'num_word_word_edges': graph['word', 'similar_to', 'word'].edge_index.shape[1],
                'num_cross_edges': graph['audio', 'transcribed_as', 'word'].edge_index.shape[1],
                'transcription_to_id': self.transcription_to_id,
                'id_to_transcription': self.id_to_transcription,
                'backbone_name': self.backbone_name,
                'representation_type': representation_type.value,
                'frame_features_shape': list(frame_features[0].shape),
                'linguistic_features_shape': list(linguistic_features[0].shape),
                'note': 'frame_features sont les features TRAMES brutes (N, T, D) - NON poolées'
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"💾 Métadonnées sauvegardées")
        
        # Vérification finale que tout est chargé
        if graph is None:
            raise RuntimeError("Le graphe n'a pas pu être chargé ou construit")
        if metadata is None:
            raise RuntimeError("Les métadonnées n'ont pas pu être chargées ou construites")
        
        # Résumé final
        print(f"\n✅ Opération terminée")
        print(f"   - frame_features: {'chargées' if frame_features is not None else 'existantes'}")
        print(f"   - linguistic_features: {'chargées' if linguistic_features is not None else 'existantes'}")
        print(f"   - graphe: {'chargé' if graph is not None else 'existant'}")
        
        return graph, metadata


# ============================================================================
# MAIN
# ============================================================================

def build_and_save_graph(config_path: str = "config/default.yaml", **kwargs):
    """
    Pipeline complet de construction du graphe.
    """
    # Charger la configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 🔑 Extraire les paramètres de la config unique
    metadata_csv = config['data']['metadata_csv']
    output_dir = config['data']['processed_dir']
    
    backbone_name = config['encoder']['backbone']
    batch_size = to_int(config['system'].get('num_workers', 8))
    graph_build = config['graph']['build']
    
    similarity_threshold = to_float(graph_build['similarity_threshold'])
    use_mst = to_bool(graph_build['use_mst'])
    backbone_method = graph_build.get('backbone_method', 'mst')
    ensure_connectivity = to_bool(graph_build.get('ensure_connectivity', True))
    
    representation_type = config['linguistic']['representation']
    device = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = to_int(config['system'].get('num_workers', 2))
    
    # Écraser avec kwargs
    for key, value in kwargs.items():
        if key == 'metadata_csv':
            metadata_csv = value
        elif key == 'output_dir':
            output_dir = value
        elif key == 'backbone':
            backbone_name = value
        elif key == 'batch_size':
            batch_size = to_int(value)
        elif key == 'threshold':
            similarity_threshold = to_float(value)
        elif key == 'no_mst':
            use_mst = not to_bool(value)
        elif key == 'representation':
            representation_type = value
        elif key == 'device':
            device = value
    
    print("="*60)
    print("CONSTRUCTION DU GRAPHE HÉTÉROGÈNE")
    print("="*60)
    print(f"📁 Metadata CSV: {metadata_csv}")
    print(f"📁 Output dir: {output_dir}")
    print(f"🔧 Backbone: {backbone_name}")
    print(f"📊 Threshold: {similarity_threshold}")
    print(f"🌳 Use MST: {use_mst}")
    print(f"💻 Device: {device}")
    
    # 1. Charger le dataset
    print("\n📂 Étape 1: Chargement du dataset")
    dataset = AudioTextDataset(
        metadata_csv=metadata_csv,
        sample_rate=config['data']['sample_rate'],
        max_length=config['data'].get('max_length', None),
        normalize=config['data'].get('normalize', True)
    )
    
    # 2. Initialiser le constructeur
    print("\n🔧 Étape 2: Initialisation du constructeur")
    builder = GraphBuilder(
        dataset=dataset,
        output_dir=output_dir,
        device=device,
        backbone_name=backbone_name
    )
    
    # 3. Configurer le graphe
    graph_config = GraphBuildConfig(
        similarity_threshold=similarity_threshold,
        use_mst=use_mst,
        backbone_method=backbone_method,
        ensure_connectivity=ensure_connectivity
    )
    
    # 4. Déterminer le type de représentation
    rep_type = LinguisticRepresentation.LABSE
    if representation_type.lower() == "labse":
        rep_type = LinguisticRepresentation.LABSE
    elif representation_type.lower() == "afriberta":
        rep_type = LinguisticRepresentation.AFRIBERTA
    elif representation_type.lower() == "bag_of_phonemes":
        rep_type = LinguisticRepresentation.BAG_OF_PHONEMES
    elif representation_type.lower() == "articulatory":
        rep_type = LinguisticRepresentation.ARTICULATORY
    else:
        print(f"⚠️ Type inconnu: {representation_type}, utilisation de LABSE")
    
    # 5. Construire et sauvegarder
    print("\n🏗️ Étape 3: Construction et sauvegarde")
    graph, metadata = builder.build_and_save_all(
        config=graph_config,
        representation_type=rep_type,
        batch_size=batch_size,
        num_workers=num_workers,
        save_features=True
    )
    
    print("\n" + "="*60)
    print("✅ FIN - Toutes les données sont prêtes!")
    print("="*60)
    print(f"\n📁 Dossier de sortie: {output_dir}")
    
    return graph, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Construction du graphe hétérogène",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Chemin vers le fichier de configuration (défaut: config/default.yaml)"
    )
    
    # Arguments optionnels (écrasent la config)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no_mst", action="store_true")
    parser.add_argument("--representation", type=str, choices=["labse", "afriberta", "bag_of_phonemes", "articulatory"], default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)
    #parser.add_argument("--pooling", type=str, choices=["mean", "max", "first"], default=None)
    
    args = parser.parse_args()
    
    # Construire les kwargs à partir des arguments non None
    kwargs = {}
    if args.metadata_csv is not None:
        kwargs['metadata_csv'] = args.metadata_csv
    if args.output_dir is not None:
        kwargs['output_dir'] = args.output_dir
    if args.backbone is not None:
        kwargs['backbone'] = args.backbone
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.threshold is not None:
        kwargs['threshold'] = args.threshold
    if args.no_mst:
        kwargs['no_mst'] = True
    if args.representation is not None:
        kwargs['representation'] = args.representation
    if args.device is not None:
        kwargs['device'] = args.device
    #if args.pooling is not None:
    #    kwargs['pooling'] = args.pooling
    
    build_and_save_graph(config_path=args.config, **kwargs)


if __name__ == "__main__":
    main()

