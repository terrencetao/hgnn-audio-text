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
EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD = 0.99


# ============================================================================
# TRAINER - FULL-BATCH
# ============================================================================

class Trainer:
    """
    Entraîneur du modèle GNN sur graphe pré-construit.
    Full-batch : tous les nœuds sont vus à chaque itération.
    """
    
    def __init__(
        self,
        config_path: str = None,
        config_override: dict = None,
        processed_dir: str = None,
        output_dir: str = None,
        train_indices: list = None,
        val_indices: list = None,
        fold: int = 0
    ):
        """
        Args:
            config_path: Chemin vers le fichier de configuration YAML
            config_override: Dictionnaire de configuration à utiliser (prioritaire)
            processed_dir: Dossier des données pré-traitées (écrase la config)
            output_dir: Dossier de sortie (écrase la config)
            train_indices: Indices pour le split d'entraînement (CV)
            val_indices: Indices pour le split de validation (CV)
            fold: Numéro du fold (pour les logs)
        """
        self.fold = fold
        
        # Charger la configuration de base
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # Écraser avec config_override si fournie
        if config_override is not None:
            self._deep_update(self.config, config_override)
        
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
        if fold > 0:
            self.output_dir = self.output_dir / f"fold_{fold}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = self.config['encoder']['backbone'].split('/')[1]
        self.representation = self.config['linguistic']['representation']
        
        # Stocker les indices pour la CV
        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # Ratio de paires négatives par rapport aux paires positives
        self.negative_ratio = self.config['training'].get('negative_ratio', 1.0)
        
        print(f"📁 Fold {self.fold}: {self.output_dir}")
        print("="*60)
        print("🏋️ INITIALISATION DU TRAINER (FULL-BATCH)")
        print("="*60)
        print(f"📁 Processed dir: {self.processed_dir}")
        print(f"📁 Output dir: {self.output_dir}")
        print(f"💻 Device: {self.device}")
        print(f"📊 Ratio négatifs/positifs: {self.negative_ratio}")
        
        # Charger les données
        self._load_graph()
        self._load_features()
        
        # Créer les masques train/test/val
        self._create_edge_masks()
        
        # Initialiser les modèles
        self._init_models()
    
    def _deep_update(self, base_dict: dict, override_dict: dict):
        """Met à jour récursivement un dictionnaire."""
        for key, value in override_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _convert_config_values(self):
        """Convertit les valeurs numériques de la config."""
        # Training
        if 'training' in self.config:
            training = self.config['training']
            for key in ['epochs', 'batch_size']:
                if key in training:
                    training[key] = to_int(training[key])
            for key in ['lr_gnn', 'weight_decay', 'dropout', 'grad_clip', 'test_edge_fraction', 'tau', 'negative_ratio']:
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
        
        # Récupérer aussi les arêtes audio-audio
        self.audio_audio_edge_index = self.graph['audio', 'similar_to', 'audio'].edge_index
        self.audio_audio_weight = self.graph['audio', 'similar_to', 'audio'].edge_weight
        
        print(f"   - Arêtes croisées: {self.cross_edge_index.shape[1]}")
        print(f"   - Arêtes audio-audio: {self.audio_audio_edge_index.shape[1]}")
    
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
        """Crée les masques pour séparer les arêtes en train/test/val."""
        test_fraction = self.config['training'].get('test_edge_fraction', 0.1)
        seed = self.config['training'].get('seed', 42) + self.fold
        
        # Créer les masques train/test pour les arêtes cross ET audio-audio
        # FIX (cf. discussion) : la restauration du RNG doit englober les
        # DEUX appels à create_test_mask -- avant, elle n'entourait que le
        # premier (cross), donc le second appel (audio-audio, seed+1000)
        # laissait le générateur global dans un état différent de
        # l'original, couplant silencieusement l'initialisation des poids
        # de _init_models() (appelée juste après) à ce second seed.
        original_state = torch.random.get_rng_state()
        
        self.train_mask, self.test_mask = create_test_mask(
            self.cross_edge_index,
            test_fraction=test_fraction,
            seed=seed
        )
        
        self.train_mask_audio, self.test_mask_audio = create_test_mask(
            self.audio_audio_edge_index,
            test_fraction=test_fraction,
            seed=seed + 1000
        )
        
        torch.random.set_rng_state(original_state)
        
        # Si des indices de validation sont fournis (CV), créer un masque de validation
        self.val_mask = None
        self.val_mask_audio = None
        
        if self.val_indices is not None:
            val_audio_mask = torch.zeros(self.frame_features.shape[0], dtype=torch.bool)
            val_audio_mask[self.val_indices] = True
            
            # Pour les arêtes cross
            audio_src = self.cross_edge_index[0]
            self.val_mask = val_audio_mask[audio_src]
            self.train_mask = self.train_mask & ~self.val_mask
            self.test_mask = self.test_mask & ~self.val_mask
            
            # Pour les arêtes audio-audio
            audio_src_audio = self.audio_audio_edge_index[0]
            audio_dst_audio = self.audio_audio_edge_index[1]
            self.val_mask_audio = val_audio_mask[audio_src_audio] | val_audio_mask[audio_dst_audio]
            self.train_mask_audio = self.train_mask_audio & ~self.val_mask_audio
            self.test_mask_audio = self.test_mask_audio & ~self.val_mask_audio
        
        # Si des indices d'entraînement sont fournis (CV), restreindre le train
        if self.train_indices is not None:
            train_audio_mask = torch.zeros(self.frame_features.shape[0], dtype=torch.bool)
            train_audio_mask[self.train_indices] = True
            
            # Pour les arêtes cross
            audio_src = self.cross_edge_index[0]
            train_mask_restricted = train_audio_mask[audio_src]
            self.train_mask = self.train_mask & train_mask_restricted
            
            # Pour les arêtes audio-audio
            audio_src_audio = self.audio_audio_edge_index[0]
            audio_dst_audio = self.audio_audio_edge_index[1]
            train_mask_audio_restricted = train_audio_mask[audio_src_audio] & train_audio_mask[audio_dst_audio]
            self.train_mask_audio = self.train_mask_audio & train_mask_audio_restricted
        
        print(f"\n🔍 Masques d'arêtes:")
        print(f"   - Arêtes cross totales: {self.cross_edge_index.shape[1]}")
        print(f"     - Train: {self.train_mask.sum().item()}")
        print(f"     - Test: {self.test_mask.sum().item()}")
        if self.val_mask is not None:
            print(f"     - Val: {self.val_mask.sum().item()}")
        
        print(f"   - Arêtes audio-audio totales: {self.audio_audio_edge_index.shape[1]}")
        print(f"     - Train: {self.train_mask_audio.sum().item()}")
        print(f"     - Test: {self.test_mask_audio.sum().item()}")
        if self.val_mask_audio is not None:
            print(f"     - Val: {self.val_mask_audio.sum().item()}")
    
    def _init_models(self):
        """Initialise les modèles."""
        hidden_dim = self.config['encoder']['hidden_dim']
        n_layers = to_int(self.config['graph']['gnn_layers'])
        dropout = to_float(self.config['training']['dropout'])
        lr = to_float(self.config['training']['lr_gnn'])
        weight_decay = to_float(self.config['training']['weight_decay'])
        
        # GNN
        self.gnn = HeterogeneousGNN(
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)
        
        # Link Predictor
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
        """Récupère les paires (audio, word) à masquer pour éviter la fuite de supervision."""
        if self.train_mask is not None and self.train_mask.sum() > 0:
            return self.cross_edge_index[:, self.train_mask]
        return torch.tensor([[], []], dtype=torch.long, device=self.device)
    
    def _sample_negatives_supervised(
        self,
        anchor_indices: torch.Tensor,
        positive_indices: torch.Tensor,
        all_embeddings: torch.Tensor,
        n_negatives_per_pair: int = 1,
    ) -> torch.Tensor:
        """Échantillonne des négatifs pour chaque paire (ancre, positif)."""
        K = anchor_indices.shape[0]
        N = all_embeddings.shape[0]
        
        neg_indices = torch.randint(0, N, (K, n_negatives_per_pair), device=self.device)
        
        for i in range(K):
            anchor_idx = anchor_indices[i].item()
            positive_idx = positive_indices[i].item()
            
            for j in range(n_negatives_per_pair):
                while neg_indices[i, j] == anchor_idx or neg_indices[i, j] == positive_idx:
                    neg_indices[i, j] = torch.randint(0, N, (1,), device=self.device).item()
        
        return all_embeddings[neg_indices]
    
    def _sample_negative_edges(
        self,
        positive_edges: torch.Tensor,
        num_nodes_src: int,
        num_nodes_dst: int,
        existing_edges: torch.Tensor,
        n_negatives: int
    ) -> torch.Tensor:
        """
        Échantillonne des paires négatives (non présentes dans existing_edges).
        
        Args:
            positive_edges: (2, E) arêtes positives
            num_nodes_src: Nombre de nœuds source
            num_nodes_dst: Nombre de nœuds destination
            existing_edges: (2, E) arêtes existantes à éviter
            n_negatives: Nombre de paires négatives à générer
        
        Returns:
            (2, n_negatives) arêtes négatives
        """
        if n_negatives <= 0:
            return torch.tensor([[], []], dtype=torch.long, device=self.device)
        
        # Créer un ensemble des arêtes existantes pour vérification rapide
        existing_set = set()
        for i in range(existing_edges.shape[1]):
            src = existing_edges[0, i].item()
            dst = existing_edges[1, i].item()
            existing_set.add((src, dst))
        
        negative_edges = []
        max_attempts = n_negatives * 10
        attempts = 0
        
        while len(negative_edges) < n_negatives and attempts < max_attempts:
            # Échantillonner aléatoirement
            src_idx = torch.randint(0, num_nodes_src, (1,)).item()
            dst_idx = torch.randint(0, num_nodes_dst, (1,)).item()
            
            # FIX (cf. discussion) : exclure les auto-boucles (src == dst).
            # Pertinent pour les négatifs audio-audio (num_nodes_src ==
            # num_nodes_dst) -- sans ça, un nœud comparé à lui-même peut
            # être accepté comme "négatif" (jamais une vraie arête
            # positive, cf. fill_diagonal_(False) dans build_audio_audio_edges),
            # ce qui pousse le link predictor à prédire "aucun lien" pour
            # un nœud comparé à lui-même -- un signal d'entraînement
            # dégénéré. Sans effet pour les négatifs cross (audio vs word,
            # jamais le même espace d'indices).
            if src_idx == dst_idx:
                attempts += 1
                continue
            
            # Vérifier que ce n'est pas une arête existante
            if (src_idx, dst_idx) not in existing_set:
                negative_edges.append([src_idx, dst_idx])
            
            attempts += 1
        
        if len(negative_edges) < n_negatives:
            print(f"   ⚠️ Impossible de générer assez de paires négatives: {len(negative_edges)}/{n_negatives}")
        
        negative_edges = torch.tensor(negative_edges, dtype=torch.long, device=self.device)
        if negative_edges.shape[0] == 0:
            return torch.tensor([[], []], dtype=torch.long, device=self.device)
        
        return negative_edges.T  # (2, n_negatives)
    
    def _forward_gnn(self, edge_index_dict: dict, return_pooled: bool = True):
        """Forward pass du GNN."""
        features_dict = {
            'audio': self.frame_features,
            'word': self.linguistic_features
        }
        attention_mask_dict = {
            'audio': None,
            'word': None
        }
        
        return self.gnn(
            features_dict=features_dict,
            attention_mask_dict=attention_mask_dict,
            edge_index_dict=edge_index_dict,
            return_pooled=return_pooled
        )
    
    def _get_train_audio_audio_edges_for_message_passing(self) -> torch.Tensor:
        """
        Arêtes audio-audio à utiliser pour le message passing du GNN --
        UNIQUEMENT celles du train (train_mask_audio), symétrisées
        explicitement.

        FIX CRITIQUE (cf. discussion) : avant, edge_index_dict utilisait
        self.graph['audio','similar_to','audio'].edge_index (le graphe
        COMPLET, non filtré) -- les arêtes de test audio-audio restaient
        donc visibles au message passing. l_test_audio était alors
        calculée sur des embeddings qui avaient déjà "vu" l'arête qu'ils
        sont censés prédire en aveugle : une fuite directe, plus flagrante
        encore que celle déjà identifiée côté texte (rev_transcribed_as).

        La symétrisation explicite est nécessaire car train_mask_audio
        opère sur des entrées individuelles de edge_index -- chaque sens
        d'une paire non-dirigée est une entrée séparée dans le tenseur, et
        rien ne garantit que (i,j) et (j,i) tombent du même côté du split
        train/test après un tirage aléatoire sur les index à plat.
        """
        train_edges = self.audio_audio_edge_index[:, self.train_mask_audio]
        return torch.cat([train_edges, train_edges.flip(0)], dim=1)
    
    def train_epoch(self) -> tuple[float, float, float]:
        """Entraîne une époque en full-batch."""
        self.gnn.train()
        self.link_predictor.train()
        
        orphan_fraction = to_float(self.config['dropout']['orphan_fraction'])
        alpha = to_float(self.config['loss_weights']['alpha'])
        beta = to_float(self.config['loss_weights']['beta'])
        grad_clip = to_float(self.config['training']['grad_clip'])
        tau = to_float(self.config['training'].get('tau', 0.1))
        negative_ratio = to_float(self.config['training'].get('negative_ratio', 1.0))
        
        self.optimizer.zero_grad()
        
        # 1. Appliquer le dropout de modalité SOFT
        orphan_mask = sample_orphan_mask(
            self.graph['audio'].num_nodes,
            orphan_fraction,
            self.device
        )
        
        masked_edge_index, masked_edge_weight = apply_soft_dropout(
            self.cross_edge_index,
            self.cross_edge_weight,
            orphan_mask,
            train_mask=self.train_mask
        )
        
        # 1bis. Anti-fuite de supervision (INDÉPENDANTE du dropout de
        # modalité, cf. discussion "Distinction clé") -- empêche un nœud
        # non-orphelin de "voir" sa propre arête cible via la relation
        # inverse ("word","rev_transcribed_as","audio") pendant
        # l'agrégation qui produit son propre embedding. Restée inactive
        # dans les versions précédentes malgré _get_target_pairs() déjà
        # implémentée -- réellement branchée ici.
        target_pairs = self._get_target_pairs()
        if target_pairs is not None and target_pairs.shape[1] > 0:
            masked_edge_index, masked_edge_weight = apply_supervision_leak_mask(
                masked_edge_index, masked_edge_weight, target_pairs
            )
        
        # 2. Préparer les arêtes du GNN
        # FIX (cf. discussion) : les arêtes audio-audio de TEST sont
        # maintenant exclues du message passing (symétrie de traitement
        # avec les arêtes croisées, qui utilisent déjà masked_edge_index
        # plutôt que le graphe complet).
        train_audio_audio_edges = self._get_train_audio_audio_edges_for_message_passing()
        edge_index_dict = {
            ('audio', 'similar_to', 'audio'): train_audio_audio_edges,
            ('word', 'similar_to', 'word'): self.graph['word', 'similar_to', 'word'].edge_index,
            ('audio', 'transcribed_as', 'word'): masked_edge_index,
            ('word', 'rev_transcribed_as', 'audio'): masked_edge_index.flip(0)
        }
        
        # 3. Forward pass du GNN
        x_dict, pooled_dict = self._forward_gnn(edge_index_dict, return_pooled=True)
        
        # 4. Adapter le Link Predictor si nécessaire
        if self.link_predictor.net[0].in_features != x_dict['audio'].shape[-1] * 2:
            hidden_dim = x_dict['audio'].shape[-1]
            self.link_predictor = LinkPredictor(hidden_dim).to(self.device)
            lr = to_float(self.config['training']['lr_gnn'])
            weight_decay = to_float(self.config['training']['weight_decay'])
            self.optimizer = torch.optim.AdamW(
                list(self.gnn.parameters()) + list(self.link_predictor.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
        
        # 5. Préparer les arêtes d'entraînement
        train_edges = self.cross_edge_index[:, self.train_mask]
        train_weights = self.cross_edge_weight[self.train_mask]
        
        train_audio_edges = self.audio_audio_edge_index[:, self.train_mask_audio]
        train_audio_weights = self.audio_audio_weight[self.train_mask_audio]
        
        # Nombre de nœuds
        n_audio = self.graph['audio'].num_nodes
        n_word = self.graph['word'].num_nodes
        
        # ====================================================================
        # 6. L_reg : Prédiction des liens audio-word (transcription)
        # ====================================================================
        l_reg_cross = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if train_edges.shape[1] > 0:
            # Paires positives
            h_audio_edges = x_dict['audio'][train_edges[0]]
            h_word_edges = x_dict['word'][train_edges[1]]
            p_pred_cross = self.link_predictor(h_audio_edges, h_word_edges)
            l_reg_cross_pos = link_regularization_loss(p_pred_cross, train_weights)
            
            # Paires négatives (audio-word qui n'ont PAS de relation)
            n_neg_cross = int(train_edges.shape[1] * negative_ratio)
            
            # Construire toutes les arêtes existantes pour éviter de les échantillonner
            all_cross_edges = self.cross_edge_index  # Toutes les arêtes cross existantes
            
            negative_cross_edges = self._sample_negative_edges(
                positive_edges=train_edges,
                num_nodes_src=n_audio,
                num_nodes_dst=n_word,
                existing_edges=all_cross_edges,
                n_negatives=n_neg_cross
            )
            
            if negative_cross_edges.shape[1] > 0:
                # Éviter les nœuds orphelins dans les négatifs
                # On vérifie que les nœuds négatifs ne sont pas orphelins
                # (mais on ne peut pas le savoir directement ici, donc on laisse le GNN gérer)
                
                h_audio_neg = x_dict['audio'][negative_cross_edges[0]]
                h_word_neg = x_dict['word'][negative_cross_edges[1]]
                p_pred_cross_neg = self.link_predictor(h_audio_neg, h_word_neg)
                
                # Les poids des négatifs sont 0 (pas de relation)
                neg_weights = torch.zeros(negative_cross_edges.shape[1], device=self.device)
                l_reg_cross_neg = link_regularization_loss(p_pred_cross_neg, neg_weights)
            else:
                l_reg_cross_neg = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            l_reg_cross = l_reg_cross_pos + l_reg_cross_neg
        else:
            l_reg_cross = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ====================================================================
        # 7. L_reg_audio : Prédiction des liens audio-audio (similarité)
        # ====================================================================
        l_reg_audio = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if train_audio_edges.shape[1] > 0:
            # Paires positives
            h_audio_src = x_dict['audio'][train_audio_edges[0]]
            h_audio_dst = x_dict['audio'][train_audio_edges[1]]
            p_pred_audio = self.link_predictor(h_audio_src, h_audio_dst)
            l_reg_audio_pos = link_regularization_loss(p_pred_audio, train_audio_weights)
            
            # Paires négatives (audio-audio qui n'ont PAS de relation)
            n_neg_audio = int(train_audio_edges.shape[1] * negative_ratio)
            
            all_audio_edges = self.audio_audio_edge_index  # Toutes les arêtes audio-audio existantes
            
            negative_audio_edges = self._sample_negative_edges(
                positive_edges=train_audio_edges,
                num_nodes_src=n_audio,
                num_nodes_dst=n_audio,
                existing_edges=all_audio_edges,
                n_negatives=n_neg_audio
            )
            
            if negative_audio_edges.shape[1] > 0:
                h_audio_src_neg = x_dict['audio'][negative_audio_edges[0]]
                h_audio_dst_neg = x_dict['audio'][negative_audio_edges[1]]
                p_pred_audio_neg = self.link_predictor(h_audio_src_neg, h_audio_dst_neg)
                
                neg_weights_audio = torch.zeros(negative_audio_edges.shape[1], device=self.device)
                l_reg_audio_neg = link_regularization_loss(p_pred_audio_neg, neg_weights_audio)
            else:
                l_reg_audio_neg = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            l_reg_audio = l_reg_audio_pos + l_reg_audio_neg
        else:
            l_reg_audio = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # L_reg totale
        l_reg = l_reg_cross + l_reg_audio
        
        # ====================================================================
        # 8. L_test pour le monitoring (sur les arêtes de test)
        # ====================================================================
        with torch.no_grad():
            # Test cross - seulement les paires positives
            if self.test_mask.sum() > 0:
                test_edges = self.cross_edge_index[:, self.test_mask]
                test_weights = self.cross_edge_weight[self.test_mask]
                
                if test_edges.shape[1] > 0:
                    h_audio_edges_test = x_dict['audio'][test_edges[0]]
                    h_word_edges_test = x_dict['word'][test_edges[1]]
                    p_pred_test_cross = self.link_predictor(h_audio_edges_test, h_word_edges_test)
                    l_test_cross = link_regularization_loss(p_pred_test_cross, test_weights)
                else:
                    l_test_cross = torch.tensor(0.0, device=self.device)
            else:
                l_test_cross = torch.tensor(0.0, device=self.device)
            
            # Test audio-audio - seulement les paires positives
            if self.test_mask_audio.sum() > 0:
                test_audio_edges = self.audio_audio_edge_index[:, self.test_mask_audio]
                test_audio_weights = self.audio_audio_weight[self.test_mask_audio]
                
                if test_audio_edges.shape[1] > 0:
                    h_audio_src_test = x_dict['audio'][test_audio_edges[0]]
                    h_audio_dst_test = x_dict['audio'][test_audio_edges[1]]
                    p_pred_test_audio = self.link_predictor(h_audio_src_test, h_audio_dst_test)
                    l_test_audio = link_regularization_loss(p_pred_test_audio, test_audio_weights)
                else:
                    l_test_audio = torch.tensor(0.0, device=self.device)
            else:
                l_test_audio = torch.tensor(0.0, device=self.device)
            
            l_test = l_test_cross + l_test_audio
        
        # ====================================================================
        # 9. L_acoustic (contrastif audio-audio)
        # ====================================================================
        if x_dict is not None and 'audio' in x_dict:
            audio_pooled = x_dict['audio']
            audio_audio_edge_index = self.graph['audio', 'similar_to', 'audio'].edge_index
            audio_audio_weight = self.graph['audio', 'similar_to', 'audio'].edge_weight
            
            positive_mask = (audio_audio_weight >= EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD)
            positive_edges = audio_audio_edge_index[:, positive_mask]
            
            if positive_edges.shape[1] > 0:
                h_negatives = self._sample_negatives_supervised(
                    anchor_indices=positive_edges[0],
                    positive_indices=positive_edges[1],
                    all_embeddings=audio_pooled,
                    n_negatives_per_pair=1
                )
                h_anchor = audio_pooled[positive_edges[0]]
                h_positive = audio_pooled[positive_edges[1]]
                l_acoustic = acoustic_loss(h_anchor, h_positive, h_negatives, tau=tau)
            else:
                l_acoustic = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            l_acoustic = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ====================================================================
        # 10. L_contrast (alignement audio-texte)
        # ====================================================================
        if (x_dict is not None and 'word' in x_dict and 'audio' in x_dict):
            h_word_pooled = x_dict['word']
            h_audio_pooled = x_dict['audio']
            
            if train_edges.shape[1] > 0:
                positive_mask = (train_weights >= EXACT_TRANSCRIPTION_WEIGHT_THRESHOLD)
                positive_edges = train_edges[:, positive_mask]
                
                if positive_edges.shape[1] > 0:
                    h_audio_neg = self._sample_negatives_supervised(
                        anchor_indices=positive_edges[1],
                        positive_indices=positive_edges[0],
                        all_embeddings=h_audio_pooled,
                        n_negatives_per_pair=1
                    )
                    h_word = h_word_pooled[positive_edges[1]]
                    h_audio_pos = h_audio_pooled[positive_edges[0]]
                    l_contrast = contrastive_alignment_loss(
                        h_word, h_audio_pos, h_audio_neg, tau=tau
                    )
                else:
                    l_contrast = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                l_contrast = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            l_contrast = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ====================================================================
        # 11. Loss totale
        # ====================================================================
        loss = total_loss(l_acoustic, l_reg, l_contrast, alpha, beta)
        
        # ====================================================================
        # 12. Backward
        # ====================================================================
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.link_predictor.parameters()),
            grad_clip
        )
        self.optimizer.step()
        
        return loss.item(), l_reg.item(), l_test.item()
    
    def train(self, epochs: int = None) -> dict:
        """Boucle d'entraînement complète."""
        print("\n" + "="*60)
        print("🏋️ DÉBUT DE L'ENTRAÎNEMENT (FULL-BATCH)")
        print("="*60)
        
        if epochs is None:
            epochs = to_int(self.config['training']['epochs'])
        
        best_loss = float('inf')
        best_test_loss = float('inf')
        best_epoch = 0
        
        model_path = self.output_dir / 'models'
        model_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            loss, reg_loss, test_loss = self.train_epoch()
            
            if torch.isnan(torch.tensor(loss)):
                print(f"⚠️ Loss NaN à l'epoch {epoch}, arrêt")
                break
            
            print(f"Epoch {epoch}/{epochs}: Loss = {loss:.4f}, Reg = {reg_loss:.4f}, Test = {test_loss:.4f}")
            
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
    
    def evaluate(self) -> dict:
        """
        Évalue le modèle entraîné sur les données.
        Utilise les mêmes modèles que ceux qui ont été entraînés.
        """
        self.gnn.eval()
        self.link_predictor.eval()
        
        with torch.no_grad():
            # Utiliser toutes les arêtes pour l'évaluation
            edge_index_dict = {
                ('audio', 'similar_to', 'audio'): self.graph['audio', 'similar_to', 'audio'].edge_index,
                ('word', 'similar_to', 'word'): self.graph['word', 'similar_to', 'word'].edge_index,
                ('audio', 'transcribed_as', 'word'): self.cross_edge_index,
                ('word', 'rev_transcribed_as', 'audio'): self.cross_edge_index.flip(0)
            }
            
            x_dict, pooled_dict = self._forward_gnn(edge_index_dict, return_pooled=True)
            
            h_audio = x_dict['audio']
            h_word = x_dict['word']
            
            # Calculer les métriques d'alignement et d'uniformité
            min_len = min(h_audio.shape[0], h_word.shape[0])
            h_audio_pos = h_audio[:min_len]
            h_word_pos = h_word[:min_len]
            h_all = h_audio
            
            from evaluation.intrinsic_metrics import alignment_uniformity_score
            metrics = alignment_uniformity_score(
                h_positive_a=h_audio_pos,
                h_positive_b=h_word_pos,
                h_all=h_all
            )
            
            alignment = metrics['alignment']
            uniformity = metrics['uniformity']
            
            # Prédictions sur les arêtes de test (cross)
            if self.test_mask.sum() > 0:
                test_edges = self.cross_edge_index[:, self.test_mask]
                test_weights = self.cross_edge_weight[self.test_mask]
                
                h_audio_edges = h_audio[test_edges[0]]
                h_word_edges = h_word[test_edges[1]]
                p_pred = self.link_predictor(h_audio_edges, h_word_edges)
                
                link_pred_outputs = (p_pred.cpu(), test_weights.cpu())
            else:
                link_pred_outputs = (torch.tensor([]), torch.tensor([]))
        
        return {
            'alignment': alignment,
            'uniformity': uniformity,
            'link_pred_outputs': link_pred_outputs,
            'x_dict': x_dict,
            'pooled_dict': pooled_dict
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
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
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
