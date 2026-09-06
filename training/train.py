"""
train.py
Entraînement du modèle GNN sur le graphe pré-construit.
Full-batch : le GNN voit tous les nœuds à chaque itération.
"""

import sys
from pathlib import Path
# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import argparse
from tqdm import tqdm

# Imports du projet
from models.gnn import HeterogeneousGNN
from models.link_predictor import LinkPredictor
from models.losses import acoustic_loss, link_regularization_loss, contrastive_alignment_loss, total_loss
from graph.dropout import (
    sample_orphan_mask, 
    apply_soft_dropout, 
    apply_supervision_leak_mask,
    create_test_mask
)


# ============================================================================
# FONCTIONS DE CONVERSION
# ============================================================================

def to_float(value):
    """Convertit une valeur en float, que ce soit un int, float ou string."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def to_int(value):
    """Convertit une valeur en int, que ce soit un int, float ou string."""
    if isinstance(value, (int, float)):
        return int(value)
    elif isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return value
    return value


def to_bool(value):
    """Convertit une valeur en bool, que ce soit un bool, int ou string."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return bool(value)
    elif isinstance(value, str):
        return value.lower() in ['true', '1', 'yes', 'on']
    return value


# Seuil pour identifier une transcription exacte (poids 1.0)
EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD = 1.0


# ============================================================================
# TRAINER - FULL-BATCH
# ============================================================================

class Trainer:
    """
    Entraîneur du modèle GNN sur graphe pré-construit.
    Full-batch : tous les nœuds sont vus à chaque itération.
    """
    
    def __init__(self, config_path: str, processed_dir: str = None, output_dir: str = None):
        """
        Args:
            config_path: Chemin vers le fichier de configuration YAML
            processed_dir: Dossier des données pré-traitées (écrase la config)
            output_dir: Dossier de sortie (écrase la config)
        """
        # Charger la configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Écraser les dossiers si fournis
        if processed_dir is not None:
            self.config['data']['processed_dir'] = processed_dir
        if output_dir is not None:
            self.config['training']['output_dir'] = output_dir
        
        # Convertir les valeurs numériques
        self._convert_config_values()
        
        # Déterminer le device
        self.device = torch.device(self.config['training']['device'])
        
        # Dossiers
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = self.config['encoder']['backbone'].split('/')[1]
        self.representation = self.config['linguistic']['representation']
        
        print("="*60)
        print("🏋️ INITIALISATION DU TRAINER (FULL-BATCH)")
        print("="*60)
        print(f"📁 Processed dir: {self.processed_dir}")
        print(f"📁 Output dir: {self.output_dir}")
        print(f"💻 Device: {self.device}")
        
        # Charger les données
        self._load_graph()
        self._load_features()
        
        # Créer les masques train/test
        self._create_edge_masks()
        
        # Initialiser les modèles
        self._init_models()
    
    def _convert_config_values(self):
        """Convertit les valeurs numériques de la config."""
        print("🔄 Conversion des valeurs de configuration...")
        
        # Training
        if 'training' in self.config:
            training = self.config['training']
            for key in ['epochs', 'batch_size']:
                if key in training:
                    training[key] = to_int(training[key])
            for key in ['lr_gnn', 'weight_decay', 'dropout', 'grad_clip', 'test_edge_fraction', 'tau']:
                if key in training:
                    training[key] = to_float(training[key])
            if 'device' in training:
                training['device'] = str(training['device'])
            if 'seed' in training:
                training['seed'] = to_int(training['seed'])
        
        # Graph
        if 'graph' in self.config:
            graph = self.config['graph']
            if 'gnn_layers' in graph:
                graph['gnn_layers'] = to_int(graph['gnn_layers'])
            if 'build' in graph and 'similarity_threshold' in graph['build']:
                graph['build']['similarity_threshold'] = to_float(graph['build']['similarity_threshold'])
        
        # Loss weights
        if 'loss_weights' in self.config:
            loss_weights = self.config['loss_weights']
            for key in ['alpha', 'beta']:
                if key in loss_weights:
                    loss_weights[key] = to_float(loss_weights[key])
        
        # Dropout
        if 'dropout' in self.config and 'orphan_fraction' in self.config['dropout']:
            self.config['dropout']['orphan_fraction'] = to_float(self.config['dropout']['orphan_fraction'])
        
        # Data
        if 'data' in self.config:
            data = self.config['data']
            for key in ['sample_rate', 'max_length']:
                if key in data:
                    data[key] = to_int(data[key])
            if 'normalize' in data:
                data['normalize'] = to_bool(data['normalize'])
        
        print("   ✅ Conversion terminée")
    
    def _load_graph(self):
        """Charge le graphe pré-construit."""
        graph_path = self.processed_dir / f'graph/hetero_graph_{self.model}_{self.representation}.pt'
        if not graph_path.exists():
            raise FileNotFoundError(f"Graphe non trouvé: {graph_path}")
        
        self.graph = torch.load(graph_path, map_location=torch.device('cpu'))
        if self.device.type == 'cuda':
            self.graph = self.graph.to(self.device)
        
        print(f"\n✅ Graphe chargé: {graph_path}")
        print(f"   - Nœuds audio: {self.graph['audio'].num_nodes}")
        print(f"   - Nœuds word: {self.graph['word'].num_nodes}")
        
        self.cross_edge_index = self.graph['audio', 'transcribed_as', 'word'].edge_index
        self.cross_edge_weight = self.graph['audio', 'transcribed_as', 'word'].edge_weight
        print(f"   - Arêtes croisées: {self.cross_edge_index.shape[1]}")
    
    def _load_features(self):
        """Charge les features pré-extraites."""
        features_dir = self.processed_dir / 'features'
        
        # Charger les features trames (audio)
        frame_path = features_dir / f'frame_features_{self.model}.pt'
        if not frame_path.exists():
            raise FileNotFoundError(f"frame_features.pt non trouvé: {frame_path}")
        self.frame_features = torch.load(frame_path, map_location=torch.device('cpu'))
        print(f"✅ Frame features: {self.frame_features.shape}")
        
        # Charger les features linguistiques
        linguistic_path = features_dir / f'linguistic_features_{self.representation}.pt'
        if not linguistic_path.exists():
            raise FileNotFoundError(f"linguistic_features.pt non trouvé: {linguistic_path}")
        self.linguistic_features = torch.load(linguistic_path, map_location=torch.device('cpu'))
        print(f"✅ Linguistic features: {self.linguistic_features.shape}")
        
        # Déplacer sur GPU si nécessaire
        if self.device.type == 'cuda':
            self.frame_features = self.frame_features.to(self.device)
            self.linguistic_features = self.linguistic_features.to(self.device)
        
        print(f"\n✅ Toutes les features chargées")
        print(f"   - Audio (trames): {self.frame_features.shape}")
        print(f"   - Linguistic (poolées): {self.linguistic_features.shape}")
    
    def _create_edge_masks(self):
        """Crée les masques pour séparer les arêtes en train/test."""
        test_fraction = self.config['training'].get('test_edge_fraction', 0.1)
        seed = self.config['training'].get('seed', 42)
        
        # Sauvegarder l'état du générateur aléatoire pour éviter le seed coupling
        original_state = torch.random.get_rng_state()
        
        self.train_mask, self.test_mask = create_test_mask(
            self.cross_edge_index,
            test_fraction=test_fraction,
            seed=seed
        )
        
        # Restaurer l'état du générateur
        torch.random.set_rng_state(original_state)
        
        print(f"\n🔍 Masques d'arêtes:")
        print(f"   - Arêtes totales: {self.cross_edge_index.shape[1]}")
        print(f"   - Arêtes train: {self.train_mask.sum().item()}")
        print(f"   - Arêtes test: {self.test_mask.sum().item()}")
    
    def _init_models(self):
        """Initialise les modèles."""
        # Récupérer les paramètres de la config
        hidden_dim = self.config['encoder']['hidden_dim']
        n_layers = to_int(self.config['graph']['gnn_layers'])
        dropout = to_float(self.config['training']['dropout'])
        lr = to_float(self.config['training']['lr_gnn'])
        weight_decay = to_float(self.config['training']['weight_decay'])
        
        # GNN avec adaptation dynamique
        self.gnn = HeterogeneousGNN(
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)
        
        # Link Predictor (dimension sera mise à jour après le premier forward)
        self.link_predictor = LinkPredictor(hidden_dim).to(self.device)
        
        # Optimiseur
        self.optimizer = torch.optim.AdamW(
            list(self.gnn.parameters()) + list(self.link_predictor.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        print(f"\n✅ Modèles initialisés (FULL-BATCH)")
        print(f"   - Hidden dim: {hidden_dim}")
        print(f"   - GNN layers: {n_layers}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Learning rate: {lr}")
        print(f"   - Weight decay: {weight_decay}")
        print(f"   - GNN: {sum(p.numel() for p in self.gnn.parameters())} paramètres")
        print(f"   - Link Predictor: {sum(p.numel() for p in self.link_predictor.parameters())} paramètres")
    
    def _get_target_pairs(self) -> torch.Tensor:
        """
        Récupère les paires (audio, word) à masquer pour éviter la fuite de supervision.
        Retourne un tenseur de shape (2, E) contenant les indices audio et word.
        """
        if self.train_mask is not None and self.train_mask.sum() > 0:
            return self.cross_edge_index[:, self.train_mask]
        return torch.tensor([[], []], dtype=torch.long, device=self.device)
    
    def _sample_negatives_supervised(
        self,
        anchor_indices: torch.Tensor,      # (K,) - indices des ancres
        positive_indices: torch.Tensor,    # (K,) - indices des positifs
        all_embeddings: torch.Tensor,      # (N, D) - tous les embeddings disponibles pour les négatifs
        n_negatives_per_pair: int = 1,     # Nombre de négatifs par paire (1 = autant de négatifs que de positifs)
    ) -> torch.Tensor:
        """
        Échantillonne des négatifs pour chaque paire (ancre, positif) en mode supervisé.
        Garantit que les négatifs sont différents de l'ancre ET du positif.
        
        Args:
            anchor_indices: (K,) - indices des ancres
            positive_indices: (K,) - indices des positifs
            all_embeddings: (N, D) - tous les embeddings disponibles pour les négatifs
            n_negatives_per_pair: nombre de négatifs par paire (1 = autant que de positifs)
        
        Returns:
            (K, n_negatives_per_pair, D) - embeddings des négatifs
        """
        K = anchor_indices.shape[0]
        N = all_embeddings.shape[0]
        
        # Générer des indices négatifs
        neg_indices = torch.randint(0, N, (K, n_negatives_per_pair), device=self.device)
        
        # S'assurer que les négatifs sont différents de l'ancre ET du positif
        for i in range(K):
            anchor_idx = anchor_indices[i].item()
            positive_idx = positive_indices[i].item()
            
            for j in range(n_negatives_per_pair):
                while neg_indices[i, j] == anchor_idx or neg_indices[i, j] == positive_idx:
                    neg_indices[i, j] = torch.randint(0, N, (1,), device=self.device).item()
        
        # Récupérer les embeddings
        return all_embeddings[neg_indices]  # (K, n_negatives_per_pair, D)
    
    def train_epoch(self) -> tuple[float, float, float]:
        """Entraîne une époque en full-batch."""
        self.gnn.train()
        self.link_predictor.train()
        
        orphan_fraction = to_float(self.config['dropout']['orphan_fraction'])
        alpha = to_float(self.config['loss_weights']['alpha'])
        beta = to_float(self.config['loss_weights']['beta'])
        grad_clip = to_float(self.config['training']['grad_clip'])
        tau = to_float(self.config['training'].get('tau', 0.1))
        
        self.optimizer.zero_grad()
        
        # 1. Appliquer le dropout de modalité SOFT
        orphan_mask = sample_orphan_mask(
            self.graph['audio'].num_nodes,
            orphan_fraction,
            self.device
        )
        
        # Appliquer le masque sur les arêtes cross
        masked_edge_index, masked_edge_weight = apply_soft_dropout(
            self.cross_edge_index,
            self.cross_edge_weight,
            orphan_mask,
            train_mask=self.train_mask
        )
        
        # Debug: monitorer le nombre de nœuds orphelins
        n_orphans = orphan_mask.sum().item()
        n_total = self.graph['audio'].num_nodes
        if n_orphans > 0:
            print(f"   🔇 Nœuds orphelins: {n_orphans}/{n_total} ({100*n_orphans/n_total:.1f}%)")
        
        # 2. Préparer les entrées du GNN
        features_dict = {
            'audio': self.frame_features,
            'word': self.linguistic_features
        }
        attention_mask_dict = {
            'audio': None,
            'word': None
        }
        
        edge_index_dict = {
            ('audio', 'similar_to', 'audio'): self.graph['audio', 'similar_to', 'audio'].edge_index,
            ('word', 'similar_to', 'word'): self.graph['word', 'similar_to', 'word'].edge_index,
            ('audio', 'transcribed_as', 'word'): masked_edge_index,
            ('word', 'rev_transcribed_as', 'audio'): masked_edge_index.flip(0)
        }
        
        # 3. Forward pass du GNN
        x_dict, pooled_dict = self.gnn(
            features_dict=features_dict,
            attention_mask_dict=attention_mask_dict,
            edge_index_dict=edge_index_dict,
            return_pooled=True
        )
        
        # 4. Adapter le Link Predictor si nécessaire
        if self.link_predictor.net[0].in_features != x_dict['audio'].shape[-1] * 2:
            hidden_dim = x_dict['audio'].shape[-1]
            self.link_predictor = LinkPredictor(hidden_dim).to(self.device)
            # Réinitialiser l'optimiseur
            lr = to_float(self.config['training']['lr_gnn'])
            weight_decay = to_float(self.config['training']['weight_decay'])
            self.optimizer = torch.optim.AdamW(
                list(self.gnn.parameters()) + list(self.link_predictor.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
            print(f"   🔄 Link Predictor adapté à la dimension: {hidden_dim}")
        
        # 5. Préparer les arêtes d'entraînement
        train_edges = self.cross_edge_index[:, self.train_mask]
        train_weights = self.cross_edge_weight[self.train_mask]
        
        # 6. Calculer L_reg sur TOUTES les arêtes d'entraînement (orphelines incluses)
        if train_edges.shape[1] > 0:
            h_audio_edges = x_dict['audio'][train_edges[0]]
            h_word_edges = x_dict['word'][train_edges[1]]
            p_pred = self.link_predictor(h_audio_edges, h_word_edges)
            l_reg = link_regularization_loss(p_pred, train_weights)
        else:
            l_reg = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 7. L_test pour le monitoring (sur les arêtes de test)
        with torch.no_grad():
            if self.test_mask.sum() > 0:
                test_edges = self.cross_edge_index[:, self.test_mask]
                test_weights = self.cross_edge_weight[self.test_mask]
                
                if test_edges.shape[1] > 0:
                    h_audio_edges_test = x_dict['audio'][test_edges[0]]
                    h_word_edges_test = x_dict['word'][test_edges[1]]
                    p_pred_test = self.link_predictor(h_audio_edges_test, h_word_edges_test)
                    l_test = link_regularization_loss(p_pred_test, test_weights)
                else:
                    l_test = torch.tensor(0.0, device=self.device)
            else:
                l_test = torch.tensor(0.0, device=self.device)
        
        # 8. Calculer L_acoustic (similarité audio-audio) - VERSION SUPERVISÉE
        # Chaque ancre audio a autant de négatifs que de positifs (1 négatif par paire)
        if x_dict is not None and 'audio' in x_dict:
            audio_pooled = x_dict['audio']  # (N, D)
            
            # Récupérer les arêtes audio-audio
            audio_audio_edge_index = self.graph['audio', 'similar_to', 'audio'].edge_index
            audio_audio_weight = self.graph['audio', 'similar_to', 'audio'].edge_weight
            
            # Filtrer pour garder seulement les paires positives (transcription exacte)
            positive_mask = (audio_audio_weight >= EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD)
            positive_edges = audio_audio_edge_index[:, positive_mask]
            
            if positive_edges.shape[1] > 0:
                # L'ancre est audio[0], le positif est audio[1]
                # Générer 1 négatif par paire (autant de négatifs que de positifs)
                h_negatives = self._sample_negatives_supervised(
                    anchor_indices=positive_edges[0],      # Audio anchor
                    positive_indices=positive_edges[1],    # Audio positive
                    all_embeddings=audio_pooled,
                    n_negatives_per_pair=1
                )  # (K, 1, D)
                
                h_anchor = audio_pooled[positive_edges[0]]  # (K, D)
                h_positive = audio_pooled[positive_edges[1]]  # (K, D)
                
                # Calculer L_acoustic avec la version supervisée
                l_acoustic = acoustic_loss(
                    h_anchor,
                    h_positive,
                    h_negatives,
                    tau=tau
                )
            else:
                l_acoustic = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            l_acoustic = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 9. Calculer L_contrast (alignement audio-texte) - VERSION SUPERVISÉE
        # Chaque mot a autant de négatifs audio que de positifs (1 négatif par paire)
        if (x_dict is not None and 
            'word' in x_dict and 
            'audio' in x_dict):
            
            h_word_pooled = x_dict['word']  # (W, D)
            h_audio_pooled = x_dict['audio']  # (N, D)

            # Utiliser toutes les arêtes d'entraînement
            if train_edges.shape[1] > 0:
                # Filtrer les paires avec transcription exacte
                positive_mask = (train_weights >= EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD)
                positive_edges = train_edges[:, positive_mask]  # [audio, word]
                
                if positive_edges.shape[1] > 0:
                    # Pour L_contrast :
                    # - L'ancre est le mot (word)
                    # - Le positif est l'audio correspondant
                    # - Les négatifs sont des audios différents
                    
                    # Générer 1 négatif audio par paire (autant que de positifs)
                    h_audio_neg = self._sample_negatives_supervised(
                        anchor_indices=positive_edges[1],    # Word anchor
                        positive_indices=positive_edges[0],  # Audio positive
                        all_embeddings=h_audio_pooled,
                        n_negatives_per_pair=1
                    )  # (K, 1, D)
                    
                    h_word = h_word_pooled[positive_edges[1]]  # (K, D)
                    h_audio_pos = h_audio_pooled[positive_edges[0]]  # (K, D)
                    
                    # Calculer L_contrast avec la version supervisée
                    l_contrast = contrastive_alignment_loss(
                        h_word,
                        h_audio_pos,
                        h_audio_neg,
                        tau=tau
                    )
                else:
                    l_contrast = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                l_contrast = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            l_contrast = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 10. Loss totale
        print(f'L_acoustic: {l_acoustic.item():.4f}, L_contrast: {l_contrast.item():.4f}, L_reg: {l_reg.item():.4f}')
        loss = total_loss(l_acoustic, l_reg, l_contrast, alpha, beta)
        
        # Vérifier que loss a bien un grad_fn
        if loss.grad_fn is None:
            raise RuntimeError("La loss n'a pas de grad_fn - impossible de faire le backward")
        
        # 11. Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.link_predictor.parameters()),
            grad_clip
        )
        self.optimizer.step()
        
        return loss.item(), l_reg.item(), l_test.item()
    
    def train(self):
        """Boucle d'entraînement complète."""
        print("\n" + "="*60)
        print("🏋️ DÉBUT DE L'ENTRAÎNEMENT (FULL-BATCH)")
        print("="*60)
        
        epochs = to_int(self.config['training']['epochs'])
        best_loss = float('inf')
        best_test_loss = float('inf')
        best_epoch = 0
        
        # Créer le dossier des modèles UNE FOIS au début
        model_path = self.output_dir / 'models'
        model_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            loss, reg_loss, test_loss = self.train_epoch()
            
            # Vérifier si loss est NaN
            if torch.isnan(torch.tensor(loss)):
                print(f"⚠️ Loss NaN détectée à l'epoch {epoch}, arrêt de l'entraînement")
                break
            
            print(f"Epoch {epoch}/{epochs}: Loss = {loss:.4f}, Reg Loss = {reg_loss:.4f}, Test Loss = {test_loss:.4f}")
            
            # Sauvegarder le meilleur modèle (selon loss train)
            if loss < best_loss:
                best_loss = loss
                best_test_loss = test_loss
                best_epoch = epoch
                
                torch.save({
                    'gnn_state_dict': self.gnn.state_dict(),
                    'link_predictor_state_dict': self.link_predictor.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'loss': loss,
                    'test_loss': test_loss,
                    'reg_loss': reg_loss,
                    'hidden_dim': self.gnn.hidden_dim
                }, model_path / f'best_model_{self.model}_{self.representation}.pt')
                print(f"   ✅ Meilleur modèle sauvegardé (loss={loss:.4f}, test={test_loss:.4f})")
        
        # Sauvegarder le modèle final
        torch.save({
            'gnn_state_dict': self.gnn.state_dict(),
            'link_predictor_state_dict': self.link_predictor.state_dict(),
            'config': self.config,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_test_loss': best_test_loss,
            'hidden_dim': self.gnn.hidden_dim
        }, model_path / f'model_final_{self.model}_{self.representation}.pt')
        
        print(f"\n💾 Modèle final sauvegardé: {model_path}")
        print(f"   - Meilleur epoch: {best_epoch}")
        print(f"   - Meilleure loss train: {best_loss:.4f}")
        print(f"   - Meilleure loss test: {best_test_loss:.4f}")
        print(f"   - Dimension cachée: {self.gnn.hidden_dim}")
        
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        
        return {
            'best_loss': best_loss,
            'best_test_loss': best_test_loss,
            'best_epoch': best_epoch,
            'model_path': str(model_path),
            'hidden_dim': self.gnn.hidden_dim
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Script principal."""
    parser = argparse.ArgumentParser(description="Entraînement du GNN")
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Chemin vers le fichier de configuration (défaut: config/default.yaml)'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default=None,
        help='Dossier des données pré-traitées (écrase la config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Dossier de sortie (écrase la config)'
    )
    args = parser.parse_args()
    
    # Vérifier si CUDA est disponible
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config['training']['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA non disponible, utilisation du CPU")
        config['training']['device'] = 'cpu'
        # Sauvegarder temporairement la config modifiée
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
        print("   ✅ Configuration mise à jour avec CPU")
    
    # Créer le trainer
    trainer = Trainer(
        config_path=args.config,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir
    )
    
    # Entraîner
    results = trainer.train()
    
    print(f"\n📊 Résultats finaux:")
    print(f"   - Best loss train: {results['best_loss']:.4f}")
    print(f"   - Best loss test: {results['best_test_loss']:.4f}")
    print(f"   - Best epoch: {results['best_epoch']}")
    print(f"   - Hidden dimension: {results['hidden_dim']}")
    print(f"   - Model saved: {results['model_path']}")


if __name__ == "__main__":
    main()
