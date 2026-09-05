"""
Dropout de modalité -- régime SOFT uniquement.

Règles importantes :
1. Le dropout est RE-TIRÉ à chaque itération/batch
2. Quand un nœud audio est orphelin, on masque TOUTES ses arêtes cross
3. Le masquage est LOCAL au nœud : les arêtes des voisins restent actives
4. Le graphe stocké n'est JAMAIS modifié en place
"""

from dataclasses import dataclass
import torch


@dataclass
class ModalityDropoutConfig:
    orphan_fraction: float = 0.3
    resample_every_iteration: bool = True
    test_edge_fraction: float = 0.1


def sample_orphan_mask(
    n_audio_nodes: int, 
    orphan_fraction: float, 
    device: torch.device,
    seed: int = None
) -> torch.Tensor:
    """
    Tire aléatoirement quels nœuds audio sont orphelins pour cette itération.
    
    Returns:
        orphan_mask: (n_audio_nodes,) booléen, True = nœud orphelin
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    return torch.rand(n_audio_nodes, device=device) < orphan_fraction


def create_test_mask(
    cross_edge_index: torch.Tensor,
    test_fraction: float = 0.1,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crée un masque pour séparer les arêtes en train/test.
    """
    E = cross_edge_index.shape[1]
    torch.manual_seed(seed)
    
    indices = torch.randperm(E)
    n_test = int(E * test_fraction)
    
    test_mask = torch.zeros(E, dtype=torch.bool)
    test_mask[indices[:n_test]] = True
    
    train_mask = ~test_mask
    
    return train_mask, test_mask


def apply_soft_dropout(
    cross_edge_index: torch.Tensor,
    cross_edge_weight: torch.Tensor,
    orphan_mask: torch.Tensor,
    train_mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Masque TOUTES les arêtes cross partant des nœuds audio orphelins.
    
    Le masquage est appliqué au niveau du nœud : si un nœud est orphelin,
    TOUTES ses arêtes cross sont masquées.
    
    IMPORTANT: Les arêtes des nœuds non-orphelins restent actives,
    même si leurs voisins sont orphelins (c'est le régime SOFT).
    
    Args:
        cross_edge_index: (2, E) -- audio_idx, word_idx
        cross_edge_weight: (E,)
        orphan_mask: (N_audio,) booléen, True = nœud orphelin
        train_mask: (E,) booléen, True = arête d'entraînement
    
    Returns:
        edge_index filtré, edge_weight filtré
    """
    audio_src = cross_edge_index[0]
    
    # Masquer les arêtes des nœuds orphelins
    keep = ~orphan_mask[audio_src]
    
    # Masquer les arêtes de test
    if train_mask is not None:
        keep = keep & train_mask
    
    return cross_edge_index[:, keep], cross_edge_weight[keep]


def get_orphan_node_weights(
    cross_edge_index: torch.Tensor,
    cross_edge_weight: torch.Tensor,
    orphan_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Version alternative : retourne les arêtes des nœuds orphelins,
    utile pour l'analyse ou le monitoring.
    """
    audio_src = cross_edge_index[0]
    orphan_edges = orphan_mask[audio_src]
    return cross_edge_index[:, orphan_edges], cross_edge_weight[orphan_edges]


class ModalityDropout:
    """
    Gestionnaire de dropout de modalité avec état.
    Permet de rééchantillonner à chaque itération et de tracker les orphelins.
    """
    
    def __init__(self, config: ModalityDropoutConfig, n_audio_nodes: int, device: torch.device):
        self.config = config
        self.n_audio_nodes = n_audio_nodes
        self.device = device
        self.orphan_mask = None
        self.current_seed = None
    
    def sample(self, seed: int = None) -> torch.Tensor:
        """Échantillonne un nouveau masque d'orphelins."""
        if seed is not None:
            self.current_seed = seed
        self.orphan_mask = sample_orphan_mask(
            self.n_audio_nodes,
            self.config.orphan_fraction,
            self.device,
            self.current_seed
        )
        return self.orphan_mask
    
    def apply(self, cross_edge_index: torch.Tensor, cross_edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applique le dropout aux arêtes cross."""
        if self.orphan_mask is None:
            raise ValueError("Il faut d'abord appeler sample()")
        return apply_soft_dropout(
            cross_edge_index,
            cross_edge_weight,
            self.orphan_mask
        )
    
    def get_orphan_indices(self) -> torch.Tensor:
        """Retourne les indices des nœuds orphelins."""
        if self.orphan_mask is None:
            return torch.tensor([], device=self.device)
        return torch.where(self.orphan_mask)[0]
    
    def get_orphan_edges(self, cross_edge_index: torch.Tensor, cross_edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne les arêtes des nœuds orphelins (pour monitoring)."""
        if self.orphan_mask is None:
            return torch.empty((2, 0), device=self.device), torch.empty(0, device=self.device)
        return get_orphan_node_weights(cross_edge_index, cross_edge_weight, self.orphan_mask)
    
    def reset(self):
        """Réinitialise le masque."""
        self.orphan_mask = None
        self.current_seed = None


def apply_supervision_leak_mask(
    cross_edge_index: torch.Tensor,
    cross_edge_weight: torch.Tensor,
    target_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hygiène anti-fuite de supervision (INDÉPENDANTE du dropout de modalité,
    cf. slide "Distinction clé") : pour toute paire (i,j) utilisée comme
    cible positive dans L_contrast à cette itération, retire SPÉCIFIQUEMENT
    cette arête (i,j) de l'agrégation qui produit h_{a_i} -- sans toucher
    aux autres arêtes de i, ni à celles de ses voisins.

    Args:
        cross_edge_index: (2, E) -- ligne 0 = indices audio, ligne 1 = indices word
        cross_edge_weight: (E,)
        target_pairs: (2, B) -- paires (audio_idx, word_idx) utilisées comme
            cibles positives dans le batch courant

    Returns:
        edge_index filtré, edge_weight filtré
    """
    if target_pairs.shape[1] == 0 or cross_edge_index.shape[1] == 0:
        return cross_edge_index, cross_edge_weight
    
    # Encodage compact des paires pour recherche efficace
    M = max(cross_edge_index[1].max().item(), target_pairs[1].max().item()) + 1
    
    # Encoder les arêtes du graphe
    edge_codes = cross_edge_index[0] * M + cross_edge_index[1]
    
    # Encoder les paires cibles
    target_codes = target_pairs[0] * M + target_pairs[1]
    
    # Convertir en set pour recherche O(1)
    if edge_codes.is_cuda:
        edge_codes_cpu = edge_codes.cpu()
        target_codes_cpu = target_codes.cpu()
    else:
        edge_codes_cpu = edge_codes
        target_codes_cpu = target_codes
    
    target_set = set(target_codes_cpu.tolist())
    
    # Créer le masque (garder les arêtes qui ne sont PAS des cibles)
    keep_mask = torch.tensor(
        [code not in target_set for code in edge_codes_cpu.tolist()],
        dtype=torch.bool,
        device=cross_edge_index.device
    )
    
    return cross_edge_index[:, keep_mask], cross_edge_weight[keep_mask]
