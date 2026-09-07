"""
retrieval.py

Évaluation retrieval sur des données jamais vues.

Le graphe de référence est une fraction du graphe complet (reference_fraction).
Les requêtes sont des fichiers audio jamais vus, et on fait du retrieval
parmi des transcriptions jamais vues.

Le seuil de similarité pour l'ajout de nœuds est le même que celui utilisé
pour la construction du graphe (config.graph_build.similarity_threshold).

CORRECTIONS apportées :
1. Tous les embeddings passent par le GNN pour être dans le même espace latent
2. test_csv est optionnel - si non fourni, exécute uniquement la baseline
3. Conservation des poids d'arêtes dans extract_reference_graph
4. Structure des résultats: {condition: {direction: {k: score}}}
5. Baseline utilise les embeddings après GNN
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import json
import warnings
warnings.filterwarnings("ignore")

from graph.build_graph import GraphBuildConfig, dtw_cosine_distance
from models.gnn import HeterogeneousGNN
from models.link_predictor import LinkPredictor
from graph.build_save_graph import GraphBuilder
from evaluation.inference_options import (
    option1_audio_only_ablation,
    option2_heterogeneous_reference,
    option3_reconnection_via_link_predictor
)
from graph.similarity import (
    LinguisticRepresentation,
    compute_labse_embeddings,
    compute_bag_of_phonemes,
    compute_articulatory_features,
)
from data.datasets import AudioTextDataset, Collator

OPTION_KEY_MAP = {
    "option1": "option1_ablation",
    "option2": "option2_reference",
    "option3": "option3_reconnection",
}

# FIX (cf. discussion) : seuil pour distinguer la transcription EXACTE
# (poids 1.0, cf. build_cross_edges) des arêtes secondaires de similarité
# (poids < 1.0). Sans ce filtre, la baseline comptait un "hit" dès qu'un
# mot linguistiquement proche apparaissait dans le top-k, pas seulement
# la vraie transcription -- une tâche plus facile que celle voulue.
EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD = 0.999


def recall_at_k(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    true_indices: torch.Tensor,
    k: int = 1
) -> float:
    """
    Recall@k pour le retrieval.

    Args:
        query_embeddings: (N_query, D) - Embeddings des audios
        candidate_embeddings: (N_candidates, D) - Embeddings des transcriptions candidates
        true_indices: (N_query,) - Index de la bonne transcription pour chaque requête
        k: Nombre de voisins à considérer
    """
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)

    sims = query_embeddings @ candidate_embeddings.T
    topk = sims.topk(k, dim=-1).indices
    hits = (topk == true_indices.unsqueeze(-1)).any(dim=-1)
    return hits.float().mean().item()


def extract_reference_graph(
    full_graph: torch.Any,  # HeteroData
    reference_fraction: float = 0.5,
    seed: int = 42
) -> torch.Any:  # HeteroData
    """
    Extrait une fraction du graphe complet pour créer le graphe de référence.
    
    On garde :
    - Les nœuds audio sélectionnés
    - Les nœuds word associés à ces nœuds audio
    - Les arêtes entre les nœuds sélectionnés
    - Les poids des arêtes (pour distinguer la vérité terrain)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_audio = full_graph["audio"].x.shape[0]

    indices = np.random.permutation(num_audio)
    split_idx = int(num_audio * reference_fraction)
    ref_audio_indices = set(indices[:split_idx].tolist())

    from torch_geometric.data import HeteroData
    reference_graph = HeteroData()

    reference_graph["audio"].x = full_graph["audio"].x

    # Filtrer les arêtes audio-audio avec conservation des poids
    audio_audio_edges = full_graph["audio", "similar_to", "audio"].edge_index
    ref_mask = torch.isin(audio_audio_edges[0], torch.tensor(list(ref_audio_indices))) & \
               torch.isin(audio_audio_edges[1], torch.tensor(list(ref_audio_indices)))
    ref_audio_edges = audio_audio_edges[:, ref_mask]
    reference_graph["audio", "similar_to", "audio"].edge_index = ref_audio_edges
    
    # Conserver les poids des arêtes audio-audio si présents
    if hasattr(full_graph["audio", "similar_to", "audio"], 'edge_weight') and \
       full_graph["audio", "similar_to", "audio"].edge_weight is not None:
        ref_audio_weights = full_graph["audio", "similar_to", "audio"].edge_weight[ref_mask]
        reference_graph["audio", "similar_to", "audio"].edge_weight = ref_audio_weights

    # Filtrer les arêtes cross avec conservation des poids
    cross_edges = full_graph["audio", "transcribed_as", "word"].edge_index
    ref_cross_mask = torch.isin(cross_edges[0], torch.tensor(list(ref_audio_indices)))
    ref_cross_edges = cross_edges[:, ref_cross_mask]

    ref_word_indices = torch.unique(ref_cross_edges[1]).tolist()
    reference_graph["word"].x = full_graph["word"].x[ref_word_indices]

    word_mapping = {old: new for new, old in enumerate(ref_word_indices)}
    new_cross_edges = torch.stack([
        ref_cross_edges[0],
        torch.tensor([word_mapping[idx.item()] for idx in ref_cross_edges[1]], dtype=torch.long)
    ])
    reference_graph["audio", "transcribed_as", "word"].edge_index = new_cross_edges
    
    # Conserver les poids des arêtes cross (1.0 = transcription exacte)
    if hasattr(full_graph["audio", "transcribed_as", "word"], 'edge_weight') and \
       full_graph["audio", "transcribed_as", "word"].edge_weight is not None:
        ref_cross_weights = full_graph["audio", "transcribed_as", "word"].edge_weight[ref_cross_mask]
        reference_graph["audio", "transcribed_as", "word"].edge_weight = ref_cross_weights

    reference_graph.ref_audio_indices = list(ref_audio_indices)
    reference_graph.ref_word_indices = ref_word_indices

    print(f"📊 Graphe de référence créé (fraction={reference_fraction}):")
    print(f"   - Nœuds audio: {reference_graph['audio'].x.shape[0]} (dont {len(ref_audio_indices)} avec arêtes)")
    print(f"   - Nœuds word: {reference_graph['word'].x.shape[0]}")

    return reference_graph


class RetrievalEvaluator:
    """Évaluation retrieval sur des données jamais vues."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        representation_type: LinguisticRepresentation = LinguisticRepresentation.LABSE,
        max_edges_per_node: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.config = OmegaConf.load(config_path)
        self.checkpoint_path = checkpoint_path
        self.representation_type = representation_type
        self.max_edges_per_node = max_edges_per_node

        # Récupérer le seuil depuis la config
        self.similarity_threshold = self.config.graph.build.similarity_threshold
        print(f"🔧 Seuil de similarité (depuis config): {self.similarity_threshold}")

        # Récupérer les noms des modèles
        self.backbone_name = self.config.encoder.backbone
        self.backbone_model_name = self.backbone_name.split('/')[1] if '/' in self.backbone_name else self.backbone_name
        self.representation_model_name = self.config.linguistic.representation

        print(f"🔧 Configuration des modèles:")
        print(f"   - Backbone audio: {self.backbone_name}")
        print(f"   - Représentation linguistique: {self.representation_model_name}")

        # Charger le builder
        dataset = AudioTextDataset(
            metadata_csv=self.config.data.metadata_csv,
            sample_rate=self.config['data']['sample_rate'],
            max_length=self.config['data'].get('max_length', None),
            normalize=self.config['data'].get('normalize', True)
        )
        self.builder = GraphBuilder(
            dataset=dataset,
            backbone_name=self.backbone_name,
            output_dir=Path(self.config.output_dir)
        )

        # Charger le graphe complet
        self.full_graph = self._load_full_graph()

        # Charger les modèles
        self.gnn, self.link_predictor = self._load_models()

        # Initialiser les données pour l'encodage linguistique
        self._init_linguistic_encoder()
        
        # Dimension de l'espace latent après GNN
        self.hidden_dim = self.gnn.hidden_dim
        print(f"🔧 Dimension de l'espace latent (après GNN): {self.hidden_dim}")

        print(f"🔧 Configuration de l'ajout de nœuds:")
        print(f"   - Seuil de similarité: {self.similarity_threshold}")
        print(f"   - Max arêtes par nœud: {self.max_edges_per_node}")

    def _warmup_gnn(self, gnn: HeterogeneousGNN) -> None:
        """
        Forward factice sur le graphe complet pour forcer l'initialisation
        des couches dynamiques du GNN.
        """
        print("🔥 Warmup du GNN (initialisation des couches dynamiques)...")
        with torch.no_grad():
            features_dict = {
                "audio": self.full_graph["audio"].x.to(self.device),
                "word": self.full_graph["word"].x.to(self.device),
            }
            edge_index_dict = {
                k: v.edge_index.to(self.device)
                for k, v in self.full_graph.edge_items()
            }
            gnn(features_dict, edge_index_dict=edge_index_dict)
        print(f"   ✅ Warmup terminé -- hidden_dim={gnn.hidden_dim}")

    def _load_models(self):
        """Charge les modèles depuis le checkpoint."""
        gnn = HeterogeneousGNN(
            n_layers=self.config.graph.gnn_layers,
            dropout=self.config.training.dropout
        ).to(self.device)

        self._warmup_gnn(gnn)

        hidden_dim = gnn.hidden_dim
        link_predictor = LinkPredictor(hidden_dim).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        gnn.load_state_dict(checkpoint['gnn_state_dict'], strict=False)
        link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])

        gnn.eval()
        link_predictor.eval()

        return gnn, link_predictor

    def _load_full_graph(self):
        """Charge le graphe complet depuis le disque."""
        graph_path = Path(self.config.output_dir) / 'graph' / f'hetero_graph_{self.backbone_model_name}_{self.representation_model_name}.pt'

        if not graph_path.exists():
            raise FileNotFoundError(f"Graphe complet non trouvé: {graph_path}")

        graph = torch.load(graph_path, map_location=self.device)
        print(f"📂 Graphe complet chargé: {graph_path}")
        print(f"   - Nœuds audio: {graph['audio'].x.shape[0]}")
        print(f"   - Nœuds word: {graph['word'].x.shape[0]}")

        return graph

    def _init_linguistic_encoder(self):
        """Initialise l'encodeur linguistique selon le type de représentation."""
        if self.representation_type == LinguisticRepresentation.LABSE:
            print(f"🔤 Utilisation de LaBSE pour l'encodage linguistique")
        elif self.representation_type == LinguisticRepresentation.BAG_OF_PHONEMES:
            print(f"🔤 Utilisation du sac de phonèmes pour l'encodage linguistique")
            self.phoneme_vocab = self.config.representation.get('phoneme_vocab', [])
        elif self.representation_type == LinguisticRepresentation.ARTICULATORY:
            print(f"🔤 Utilisation des features articulatoires pour l'encodage linguistique")

    def encode_transcriptions(
        self,
        transcriptions: List[str],
        phrases_phonemes: Optional[List[List[List[str]]]] = None
    ) -> torch.Tensor:
        """Encode les transcriptions avec le modèle linguistique approprié."""
        if self.representation_type == LinguisticRepresentation.LABSE:
            return compute_labse_embeddings(transcriptions)
        elif self.representation_type == LinguisticRepresentation.BAG_OF_PHONEMES:
            if phrases_phonemes is None:
                raise ValueError("phrases_phonemes requis pour BAG_OF_PHONEMES")
            return compute_bag_of_phonemes(phrases_phonemes, self.phoneme_vocab)
        elif self.representation_type == LinguisticRepresentation.ARTICULATORY:
            if phrases_phonemes is None:
                raise ValueError("phrases_phonemes requis pour ARTICULATORY")
            features = compute_articulatory_features(phrases_phonemes, mode='mean')
            return torch.stack(features)
        else:
            raise ValueError(f"Type non supporté: {self.representation_type}")
            
    
    def build_text_audio_edges_by_similarity(
        self,
        text_embedding: torch.Tensor,
        reference_graph: torch.Any,
        max_edges: int = None
    ) -> torch.Tensor:
        """
        Construit les arêtes entre un nouveau texte et les audios du graphe de référence
        en utilisant le seuil de similarité de la config.
        """
        if max_edges is None:
            max_edges = self.max_edges_per_node

        # Embeddings des audios de référence dans l'espace latent
        ref_audio_features = reference_graph["audio"].x
        
        # Normaliser
        text_norm = F.normalize(text_embedding, p=2, dim=-1)
        ref_norm = F.normalize(ref_audio_features, p=2, dim=-1)
        
        # Similarités
        sims = text_norm @ ref_norm.T
        sims = sims.squeeze(0)
        
        # Seuil
        mask = sims > self.similarity_threshold
        valid_indices = torch.where(mask)[0]
        
        if len(valid_indices) == 0:
            best_idx = sims.argmax().item()
            valid_indices = torch.tensor([best_idx])
            print(f"   ⚠️ Aucune similarité > {self.similarity_threshold}, connexion au plus proche (sim={sims[best_idx]:.3f})")
        else:
            sorted_indices = valid_indices[torch.argsort(sims[valid_indices], descending=True)]
            if len(sorted_indices) > max_edges:
                valid_indices = sorted_indices[:max_edges]
            else:
                valid_indices = sorted_indices
        
        # Créer les arêtes (sens word→audio)
        # Le nouveau nœud word sera à la position num_ref_words
        edges = []
        for audio_idx in valid_indices:
            edges.append([audio_idx.item(), -1])  # audio → word (arête transcribed_as)
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).T
        else:
            return torch.tensor([[], []], dtype=torch.long)
    
    def compute_audio_dtw_similarity(
    query_features: torch.Tensor,
        reference_graph: torch.Any,
        )
        if frame_features.dim() != 3:
        raise ValueError(
            f"Expected (N, T, D), got {frame_features.shape}"
        )

        N = frame_features.shape[0]

        distance_matrix = torch.zeros(
            (1, N),
            dtype=torch.float32
        )

        for i in range(N):

            distance = dtw_cosine_distance(
                    query_features,
                    frame_features[j]
                )

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

        # Distance -> similarity
        similarity_matrix = torch.exp(
            -distance_matrix
        )

        return similarity_matrix
        
    def build_edges_by_similarity_threshold(
        self,
        query_features: torch.Tensor,
        reference_graph: torch.Any,  # HeteroData
        max_edges: int = None
    ) -> torch.Tensor:
        """
        Construit les arêtes entre la requête et le graphe de référence
        en utilisant le seuil de similarité de la config.
        """
        if max_edges is None:
            max_edges = self.max_edges_per_node

        ref_audio_features = reference_graph["audio"].x

        query_feats = F.normalize(query_features, p=2, dim=-1)
        ref_feats = F.normalize(ref_audio_features, p=2, dim=-1)

        sims = compute_audio_dtw_similarity(query_feats, ref_feats)
        sims = sims.squeeze(0)

        mask = sims > self.similarity_threshold
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) == 0:
            best_idx = sims.argmax().item()
            valid_indices = torch.tensor([best_idx])
            print(f"   ⚠️ Aucune similarité > {self.similarity_threshold}, connexion au plus proche (sim={sims[best_idx]:.3f})")
        else:
            sorted_indices = valid_indices[torch.argsort(sims[valid_indices], descending=True)]
            #if len(sorted_indices) > max_edges:
            #    valid_indices = sorted_indices[:max_edges]              ne pas limiter les connections
            #else:
            #    valid_indices = sorted_indices

        edges = []
        num_ref = ref_audio_features.shape[0]
        node_idx = num_ref

        for ref_idx in valid_indices:
            edges.append([node_idx, ref_idx.item()])
            edges.append([ref_idx.item(), node_idx])

        if edges:
            return torch.tensor(edges, dtype=torch.long).T
        else:
            return torch.tensor([[], []], dtype=torch.long)

    def infer_audio(
        self,
        audio_features: torch.Tensor,
        reference_graph: torch.Any,
        option: str = "option2",
        confidence_threshold: float = 0.6,
        top_k_candidates: int = 100,
        edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Infère la représentation d'un audio via l'option choisie.
        """
        if edges is None:
            edges = self.build_edges_by_similarity_threshold(audio_features, reference_graph)

        with torch.no_grad():
            if option == "option1":
                return option1_audio_only_ablation(
                    audio_features.to(self.device),
                    reference_graph[("audio", "similar_to", "audio")].edge_index.to(self.device),
                    self.gnn,
                    new_node_audio_audio_edges=edges.to(self.device),
                    reference_audio_features=reference_graph["audio"].x.to(self.device),
                )
            elif option == "option2":
                return option2_heterogeneous_reference(
                    audio_features.to(self.device),
                    reference_graph,
                    self.gnn,
                    edges.to(self.device)
                )
            elif option == "option3":
                return option3_reconnection_via_link_predictor(
                    audio_features.to(self.device),
                    reference_graph,
                    self.gnn,
                    self.link_predictor,
                    edges.to(self.device),
                    confidence_threshold=confidence_threshold,
                    top_k_candidates=top_k_candidates
                )
            else:
                raise ValueError(f"Option inconnue: {option}")

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        k: int = 1
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve les k candidats les plus proches.
        Les embeddings doivent déjà être dans le même espace latent.
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)

        sims = query_embedding @ candidate_embeddings.T
        scores, indices = sims.topk(k, dim=-1)

        return indices[0].tolist(), scores[0].tolist()

    def get_phoneme_segmentation(self, transcriptions: List[str]) -> List[List[List[str]]]:
        """Segmentation phonémique simplifiée."""
        return [[[c for c in word] for word in phrase.split()] for phrase in transcriptions]

    def _get_gnn_embeddings(self, graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtient tous les embeddings audio et texte après passage dans le GNN.
        """
        with torch.no_grad():
            features_dict = {
                "audio": graph["audio"].x.to(self.device),
                "word": graph["word"].x.to(self.device),
            }
            edge_index_dict = {
                k: v.edge_index.to(self.device)
                for k, v in graph.edge_items()
            }
            out = self.gnn(features_dict, edge_index_dict=edge_index_dict)
            return out["audio"], out["word"]

    def evaluate_baseline_only(
        self,
        reference_fraction: float = 0.5,
        k_values: List[int] = [1, 5, 10],
        seed: int = 42
    ) -> Dict:
        """
        Mode baseline uniquement: évalue le retrieval sur les nœuds du graphe de référence.
        """
        reference_graph = extract_reference_graph(
            self.full_graph,
            reference_fraction,
            seed
        )

        print("🔧 Extraction des embeddings avec le GNN...")
        audio_embeddings, word_embeddings = self._get_gnn_embeddings(reference_graph)

        results = {
            "baseline": {
                "audio_to_text": {k: [] for k in k_values},
                "text_to_audio": {k: [] for k in k_values}
            }
        }

        # Construire les mappings audio<->texte
        # FIX : ne garder que les arêtes de transcription EXACTE (poids
        # ~1.0) -- sans ça, un mot "similaire" (arête secondaire, poids <
        # 1.0) comptait aussi comme une bonne réponse, ce qui n'est pas la
        # vérité terrain voulue pour la baseline.
        audio_to_text_mapping = {}
        word_to_audio_mapping = {}
        edge_index = reference_graph["audio", "transcribed_as", "word"].edge_index
        edge_weight = reference_graph["audio", "transcribed_as", "word"].edge_weight
        exact_mask = edge_weight > EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD
        for i in range(edge_index.shape[1]):
            if not exact_mask[i]:
                continue
            audio_idx = edge_index[0, i].item()
            word_idx = edge_index[1, i].item()
            audio_to_text_mapping.setdefault(audio_idx, []).append(word_idx)
            word_to_audio_mapping.setdefault(word_idx, []).append(audio_idx)

        # Audio-to-text baseline
        print("📊 Baseline audio→text...")
        for audio_idx in tqdm(range(audio_embeddings.shape[0]), desc="Audio→Text"):
            if audio_idx not in audio_to_text_mapping:
                for k in k_values:
                    results["baseline"]["audio_to_text"][k].append(0.0)
                continue

            audio_embedding = audio_embeddings[audio_idx]
            true_word_indices = audio_to_text_mapping[audio_idx]

            for k in k_values:
                indices, _ = self.retrieve(audio_embedding, word_embeddings, k=k)
                is_correct = any(idx in indices for idx in true_word_indices)
                results["baseline"]["audio_to_text"][k].append(1.0 if is_correct else 0.0)

        # Text-to-audio baseline
        print("📊 Baseline text→audio...")
        for word_idx in tqdm(range(word_embeddings.shape[0]), desc="Text→Audio"):
            if word_idx not in word_to_audio_mapping:
                for k in k_values:
                    results["baseline"]["text_to_audio"][k].append(0.0)
                continue

            word_embedding = word_embeddings[word_idx]
            true_audio_indices = word_to_audio_mapping[word_idx]

            if word_embedding.dim() == 1:
                word_embedding = word_embedding.unsqueeze(0)

            word_norm = F.normalize(word_embedding, p=2, dim=-1)
            audio_norm = F.normalize(audio_embeddings, p=2, dim=-1)

            scores = word_norm @ audio_norm.T
            top_k_indices = scores.topk(max(k_values)).indices[0]

            for k in k_values:
                is_correct = any(idx in top_k_indices[:k].tolist() for idx in true_audio_indices)
                results["baseline"]["text_to_audio"][k].append(1.0 if is_correct else 0.0)

        # Moyennes finales
        final_results = {}
        for condition in results:
            final_results[condition] = {}
            for direction in results[condition]:
                final_results[condition][direction] = {}
                for k in k_values:
                    scores = results[condition][direction][k]
                    final_results[condition][direction][k] = np.mean(scores) if scores else 0.0

        final_results["metadata"] = {
            "reference_fraction": reference_fraction,
            "num_reference_nodes": audio_embeddings.shape[0],
            "num_reference_words": word_embeddings.shape[0],
            "k_values": k_values,
            "seed": seed,
            "mode": "baseline_only",
            "representation_type": self.representation_type.value,
            "backbone_name": self.backbone_name,
            "similarity_threshold": self.similarity_threshold,
            "max_edges_per_node": self.max_edges_per_node,
            "hidden_dim": self.hidden_dim
        }

        return final_results

    def evaluate_on_test_set(
        self,
        reference_fraction: float = 0.5,
        test_csv_path: str = None,
        k_values: List[int] = [1, 5, 10],
        seed: int = 42
    ) -> Dict:
        """
        Évalue le retrieval sur un dataset de test.
        Si test_csv_path est None, exécute uniquement la baseline.
        """
        if test_csv_path is None:
            print("🔍 Mode baseline uniquement (pas de fichier de test fourni)")
            return self.evaluate_baseline_only(reference_fraction, k_values, seed)

        test_df = pd.read_csv(test_csv_path)
        test_transcriptions = test_df['transcription'].tolist()

        # ================================================================
        # 1. Création du graphe de référence
        # ================================================================
        reference_graph = extract_reference_graph(
            self.full_graph,
            reference_fraction,
            seed
        )

        # ================================================================
        # 2. Extraction des features audio de test
        # ================================================================
        print("🎵 Extraction des features audio de test...")
        test_audio_features = []
        for idx, row in test_df.iterrows():
            try:
                audio_feat = self.builder.extract_audio_features_from_path(str(row['audio_path']))
                if audio_feat.dim() == 1:
                    # NOTE (cf. discussion) : ce check semble contredire le
                    # reste du fichier -- partout ailleurs (retrieve(),
                    # option1/2/3...), un embedding 1D (D,) est le format
                    # NORMAL, géré via `.unsqueeze(0)`. Activer un `raise`
                    # ici casserait le chemin normal si
                    # extract_audio_features_from_path retourne du 1D (ce
                    # qui semble être le cas). Je laisse donc CE check
                    # inactif intentionnellement plutôt que de l'activer à
                    # l'aveugle -- à supprimer ou corriger toi-même une
                    # fois vérifié ce que retourne réellement cette méthode.
                    pass
                test_audio_features.append(audio_feat)
            except Exception as e:
                print(f"⚠️ Erreur pour {row['audio_path']}: {e}")
                test_audio_features.append(None)

        # ================================================================
        # 3. Encodage des transcriptions de test
        # ================================================================
        print("🔤 Encodage des transcriptions de test...")
        phrases_phonemes = None
        if self.representation_type in [LinguisticRepresentation.BAG_OF_PHONEMES,
                                        LinguisticRepresentation.ARTICULATORY]:
            phrases_phonemes = self.get_phoneme_segmentation(test_transcriptions)
        test_text_features_raw = self.encode_transcriptions(test_transcriptions, phrases_phonemes)

        # ================================================================
        # 4. Filtrer les éléments valides
        # ================================================================
        valid_indices = [i for i, f in enumerate(test_audio_features) if f is not None]
        if not valid_indices:
            print("❌ Aucun audio de test valide")
            return {}
        
        print(f"📊 {len(valid_indices)} audios valides sur {len(test_df)}")

        # ================================================================
        # 5. Ajout des textes de test au graphe (AVEC arêtes word-word)
        # ================================================================
        print("🔧 Ajout des textes de test au graphe (avec arêtes word-word par similarité)...")
        
        # Construire les arêtes pour chaque texte de test
        all_text_edges = []
        text_node_offset = reference_graph["word"].x.shape[0]  # Position du premier texte de test
        
        for i, text_feat in enumerate(test_text_features_raw):
            # Calculer les arêtes par similarité avec les textes de référence
            text_edges = self._build_text_edges_by_similarity(
                text_feat.unsqueeze(0) if text_feat.dim() == 1 else text_feat,
                reference_graph
            )
            
            if text_edges is not None and text_edges.shape[1] > 0:
                # Ajuster les indices : le nouveau nœud est à la position text_node_offset + i
                text_edges[1] = torch.full_like(text_edges[1], fill_value=text_node_offset + i)
                all_text_edges.append(text_edges)
        
        # Concaténer toutes les arêtes des textes de test
        if all_text_edges:
            all_text_edges = torch.cat(all_text_edges, dim=1)
        else:
            all_text_edges = torch.tensor([[], []], dtype=torch.long)
        
        # Passer les textes de test par le GNN AVEC leurs arêtes word-word
        print("🔧 Passage des textes de test par le GNN (avec arêtes word-word)...")
        with torch.no_grad():
            temp_features_dict = {
                "audio": reference_graph["audio"].x.to(self.device),
                "word": torch.cat([
                    reference_graph["word"].x.to(self.device),
                    test_text_features_raw.to(self.device)
                ], dim=0)
            }
            
            # Ajouter les arêtes word-word si elles existent déjà dans le graphe de référence
            # (par exemple "similar_to" ou autre relation word-word)
            temp_edge_index_dict = {
                ("audio", "similar_to", "audio"): reference_graph[("audio", "similar_to", "audio")].edge_index.to(self.device),
                ("audio", "transcribed_as", "word"): reference_graph[("audio", "transcribed_as", "word")].edge_index.to(self.device),
            }
            
            # Si une relation word-word existe dans le graphe de référence, on l'ajoute
            if ("word", "similar_to", "word") in reference_graph.edge_index_dict:
                temp_edge_index_dict[("word", "similar_to", "word")] = torch.cat([
                    reference_graph[("word", "similar_to", "word")].edge_index.to(self.device),
                    all_text_edges.to(self.device)
                ], dim=1)
            else:
                # Si pas de relation word-word existante, on crée la relation
                temp_edge_index_dict[("word", "similar_to", "word")] = all_text_edges.to(self.device)
            
            temp_out = self.gnn(temp_features_dict, edge_index_dict=temp_edge_index_dict)
            
            num_ref_words = reference_graph["word"].x.shape[0]
            text_embeddings = temp_out["word"][num_ref_words:]  # Embeddings des textes de test

        # ================================================================
        # 6. Baseline sur le graphe de référence
        # ================================================================
        print("📊 Baseline sur le graphe de référence...")
        ref_audio_embeddings, ref_word_embeddings = self._get_gnn_embeddings(reference_graph)

        # Structure des résultats
        results = {
            "baseline": {
                "audio_to_text": {k: [] for k in k_values},
                "text_to_audio": {k: [] for k in k_values}
            },
            "option1_ablation": {
                "audio_to_text": {k: [] for k in k_values},
                "text_to_audio": {k: [] for k in k_values}
            },
            "option2_reference": {
                "audio_to_text": {k: [] for k in k_values},
                "text_to_audio": {k: [] for k in k_values}
            },
            "option3_reconnection": {
                "audio_to_text": {k: [] for k in k_values},
                "text_to_audio": {k: [] for k in k_values}
            }
        }

        # Baseline audio-to-text
        # FIX : même filtrage par transcription exacte que dans
        # evaluate_baseline_only (cf. discussion) -- sans ça, un mot
        # "similaire" (arête secondaire) comptait aussi comme bonne réponse.
        audio_to_text_mapping = {}
        edge_index = reference_graph["audio", "transcribed_as", "word"].edge_index
        edge_weight = reference_graph["audio", "transcribed_as", "word"].edge_weight
        exact_mask = edge_weight > EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD
        for i in range(edge_index.shape[1]):
            if not exact_mask[i]:
                continue
            audio_idx = edge_index[0, i].item()
            word_idx = edge_index[1, i].item()
            audio_to_text_mapping.setdefault(audio_idx, []).append(word_idx)

        for audio_idx in range(ref_audio_embeddings.shape[0]):
            if audio_idx not in audio_to_text_mapping:
                for k in k_values:
                    results["baseline"]["audio_to_text"][k].append(0.0)
                continue
            
            audio_embedding = ref_audio_embeddings[audio_idx]
            true_word_indices = audio_to_text_mapping[audio_idx]
            
            for k in k_values:
                indices, _ = self.retrieve(audio_embedding, ref_word_embeddings, k=k)
                is_correct = any(idx in indices for idx in true_word_indices)
                results["baseline"]["audio_to_text"][k].append(1.0 if is_correct else 0.0)

        # Baseline text-to-audio
        # FIX : même filtrage (réutilise exact_mask calculé ci-dessus)
        word_to_audio_mapping = {}
        for i in range(edge_index.shape[1]):
            if not exact_mask[i]:
                continue
            audio_idx = edge_index[0, i].item()
            word_idx = edge_index[1, i].item()
            word_to_audio_mapping.setdefault(word_idx, []).append(audio_idx)

        for word_idx in range(ref_word_embeddings.shape[0]):
            if word_idx not in word_to_audio_mapping:
                for k in k_values:
                    results["baseline"]["text_to_audio"][k].append(0.0)
                continue
            
            word_embedding = ref_word_embeddings[word_idx]
            true_audio_indices = word_to_audio_mapping[word_idx]
            
            if word_embedding.dim() == 1:
                word_embedding = word_embedding.unsqueeze(0)
            
            scores = F.cosine_similarity(word_embedding, ref_audio_embeddings, dim=-1)
            top_k_indices = scores.topk(max(k_values)).indices
            
            for k in k_values:
                is_correct = any(idx in top_k_indices[:k].tolist() for idx in true_audio_indices)
                results["baseline"]["text_to_audio"][k].append(1.0 if is_correct else 0.0)

        # ================================================================
        # 7. Options d'inférence (Audio→Text ET Text→Audio)
        # ================================================================
        print("🎵 Traitement des options d'inférence...")
        connection_stats = {"num_edges": []}

        # Stocker les embeddings de tous les audios de test pour le text-to-audio
        all_option_embeddings = {
            "option1_ablation": [],
            "option2_reference": [],
            "option3_reconnection": []
        }
        # FIX (cf. discussion) : on garde, EN PARALLÈLE de chaque embedding
        # stocké, la position réelle `i` (dans l'énumération de
        # valid_indices) à laquelle il correspond. Sans ça, filtrer les
        # échecs (None) plus bas décale les index du tenseur empilé par
        # rapport à `i`, et TOUTES les comparaisons text->audio suivantes
        # pour cette option deviennent silencieusement fausses dès le
        # premier échec rencontré.
        all_option_positions = {
            "option1_ablation": [],
            "option2_reference": [],
            "option3_reconnection": []
        }

        for i, idx in enumerate(tqdm(valid_indices, desc="Traitement")):
            audio_features = test_audio_features[idx]
            correct_transcription = test_df.iloc[idx]['transcription']

            # Ajouter le nœud audio au graphe (calcul des arêtes par similarité avec les audios de référence)
            edges = self.build_edges_by_similarity_threshold(audio_features, reference_graph)
            connection_stats["num_edges"].append(edges.shape[1] // 2)

            for option in ["option1", "option2", "option3"]:
                option_key = OPTION_KEY_MAP[option]
                try:
                    embedding = self.infer_audio(
                        audio_features,
                        reference_graph,
                        option=option,
                        edges=edges
                    )

                    # ========================================================
                    # Audio → Text: comparer l'embedding de l'audio avec les textes de test
                    # ========================================================
                    for k in k_values:
                        indices, _ = self.retrieve(embedding, text_embeddings, k=k)
                        predicted_texts = [test_transcriptions[i] for i in indices if i < len(test_transcriptions)]
                        is_correct = correct_transcription in predicted_texts
                        results[option_key]["audio_to_text"][k].append(1.0 if is_correct else 0.0)

                    # FIX : stocker l'embedding ET sa position réelle `i` ensemble
                    all_option_embeddings[option_key].append(embedding.cpu())
                    all_option_positions[option_key].append(i)

                except Exception as e:
                    print(f"⚠️ Erreur pour option {option} sur idx {idx}: {e}")
                    for k in k_values:
                        results[option_key]["audio_to_text"][k].append(0.0)
                    # FIX : ne RIEN ajouter à all_option_embeddings/positions
                    # -- pas de placeholder None à filtrer plus tard, donc
                    # plus de risque de désalignement.

        # ================================================================
        # 8. Text → Audio: Utiliser les embeddings déjà calculés
        # ================================================================
        print("📊 Text-to-Audio pour les options...")

        # Empiler les embeddings valides -- all_option_positions[option_key][row]
        # donne la position `i` réelle correspondant à la ligne `row` du
        # tenseur empilé (FIX : plus d'hypothèse d'alignement implicite).
        for option_key in all_option_embeddings:
            if all_option_embeddings[option_key]:
                all_option_embeddings[option_key] = torch.stack(all_option_embeddings[option_key])
            else:
                all_option_embeddings[option_key] = None

        for i, idx in enumerate(valid_indices):
            correct_transcription = test_df.iloc[idx]['transcription']
            correct_text_idx = test_transcriptions.index(correct_transcription) if correct_transcription in test_transcriptions else -1

            if correct_text_idx == -1:
                continue

            # Embedding du texte correct
            text_embedding = text_embeddings[correct_text_idx]
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)

            text_norm = F.normalize(text_embedding, p=2, dim=-1)

            for option_key in all_option_embeddings:
                if all_option_embeddings[option_key] is None:
                    continue

                # Embeddings de tous les audios pour cette option
                audio_embeddings = all_option_embeddings[option_key]
                audio_norm = F.normalize(audio_embeddings, p=2, dim=-1)

                # Similarité entre le texte et tous les audios
                scores = text_norm @ audio_norm.T
                top_k_rows = scores.topk(min(max(k_values), audio_norm.shape[0])).indices[0]

                # FIX : traduire les indices de LIGNE (dans le tenseur
                # empilé, potentiellement plus petit que valid_indices si
                # des échecs ont été exclus) vers les positions RÉELLES `i`
                # via all_option_positions, avant de comparer.
                positions_for_option = all_option_positions[option_key]
                top_k_true_positions = [positions_for_option[row.item()] for row in top_k_rows]

                for k in k_values:
                    is_correct = i in top_k_true_positions[:k]
                    results[option_key]["text_to_audio"][k].append(1.0 if is_correct else 0.0)

        # ================================================================
        # 9. Calculer les moyennes finales
        # ================================================================
        final_results = {}
        for condition in results:
            final_results[condition] = {}
            for direction in results[condition]:
                final_results[condition][direction] = {}
                for k in k_values:
                    scores = results[condition][direction][k]
                    final_results[condition][direction][k] = np.mean(scores) if scores else 0.0

        final_results["metadata"] = {
            "reference_fraction": reference_fraction,
            "num_reference_nodes": ref_audio_embeddings.shape[0],
            "num_reference_words": ref_word_embeddings.shape[0],
            "num_test_samples": len(valid_indices),
            "k_values": k_values,
            "seed": seed,
            "test_csv": test_csv_path,
            "representation_type": self.representation_type.value,
            "backbone_name": self.backbone_name,
            "similarity_threshold": self.similarity_threshold,
            "max_edges_per_node": self.max_edges_per_node,
            "hidden_dim": self.hidden_dim,
            "avg_edges_per_query": np.mean(connection_stats["num_edges"]) if connection_stats["num_edges"] else 0
        }

        return final_results


    def _build_text_edges_by_similarity(
        self,
        text_embedding: torch.Tensor,
        reference_graph: torch.Any,
        max_edges: int = None
    ) -> torch.Tensor:
        """
        Construit les arêtes entre un nouveau texte et les textes du graphe de référence
        en utilisant le seuil de similarité de la config.
        
        Retourne un tensor (2, E) où chaque arête est [word_idx, -1] 
        (-1 sera remplacé par l'index réel du nouveau nœud word)
        """
        if max_edges is None:
            max_edges = self.max_edges_per_node

        # Embeddings des textes de référence
        ref_word_features = reference_graph["word"].x
        
        # Normaliser
        text_norm = F.normalize(text_embedding, p=2, dim=-1)
        ref_norm = F.normalize(ref_word_features, p=2, dim=-1)
        
        # Similarités
        sims = text_norm @ ref_norm.T
        sims = sims.squeeze(0)
        
        # Seuil
        mask = sims > self.similarity_threshold
        valid_indices = torch.where(mask)[0]
        
        if len(valid_indices) == 0:
            best_idx = sims.argmax().item()
            valid_indices = torch.tensor([best_idx])
            print(f"   ⚠️ Aucune similarité > {self.similarity_threshold}, connexion au plus proche (sim={sims[best_idx]:.3f})")
        else:
            sorted_indices = valid_indices[torch.argsort(sims[valid_indices], descending=True)]
            if len(sorted_indices) > max_edges:
                valid_indices = sorted_indices[:max_edges]
            else:
                valid_indices = sorted_indices
        
        # Créer les arêtes (sens word→word)
        # Le nouveau nœud word aura l'index -1 (sera remplacé après)
        edges = []
        for ref_word_idx in valid_indices:
            edges.append([ref_word_idx.item(), -1])  # word référence → nouveau word
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).T
        else:
            return torch.tensor([[], []], dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Évaluation retrieval sur données jamais vues")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--reference_fraction", type=float, default=0.5)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--representation_type", type=str, default="labse",
                       choices=["labse", "bag_of_phonemes", "articulatory"])
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    rep_type = LinguisticRepresentation(args.representation_type)

    evaluator = RetrievalEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        representation_type=rep_type,
        max_edges_per_node=args.max_edges_per_node
    )

    results = evaluator.evaluate_on_test_set(
        reference_fraction=args.reference_fraction,
        test_csv_path=args.test_csv,
        k_values=args.k_values,
        seed=args.seed
    )

    output_dir = Path(args.output_dir) / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = evaluator.backbone_model_name
    rep_name = args.representation_type
    threshold = evaluator.similarity_threshold
    exp_name = f"retrieval_{backbone_name}_{rep_name}_f{args.reference_fraction}_thr{threshold}_max{args.max_edges_per_node}_seed{args.seed}"
    if args.test_csv is None:
        exp_name += "_baseline_only"

    # Sauvegarde en PT
    torch.save(results, output_dir / f"{exp_name}.pt")

    # Sauvegarde en JSON
    json_data = {}
    for key, value in results.items():
        if key == "metadata":
            json_data[key] = value
        elif isinstance(value, dict):
            json_data[key] = {}
            for direction, scores in value.items():
                json_data[key][direction] = {str(k): float(v) for k, v in scores.items()}
        else:
            json_data[key] = value

    with open(output_dir / f"{exp_name}.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n📊 Résultats sauvegardés dans {output_dir}")
    print(f"   - {exp_name}.pt")
    print(f"   - {exp_name}.json")


if __name__ == "__main__":
    main()
