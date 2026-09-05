"""
Les trois pertes d'entraînement (cf. slide "Rappel : les trois pertes
d'entraînement").

alpha, beta pondèrent respectivement L_acoustic et L_reg -- ce sont des
hyperparamètres À NE JAMAIS sélectionner sur la valeur finale de la loss
combinée (cf. discussion "pourquoi pas la loss globale" : un alpha petit
minimise mécaniquement sa contribution sans refléter une meilleure
représentation). La sélection passe par training/cross_validation.py.
"""

import torch
import torch.nn.functional as F


def acoustic_loss(h_anchor: torch.Tensor, h_positive: torch.Tensor, h_negatives: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    L_acoustic -- similarité locale intra-modale (audio-audio uniquement).

    Args:
        h_anchor: (K, D)
        h_positive: (K, D) -- positif correspondant à chaque ancre
        h_negatives: (K, M, D) -- M négatifs par ancre
        tau: température
    """
    pos_sim = (h_anchor * h_positive).sum(-1) / tau  # (K,)
    neg_sim = torch.einsum("kd,kmd->km", h_anchor, h_negatives) / tau  # (K, M)
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (K, 1+M)
    labels = torch.zeros(h_anchor.size(0), dtype=torch.long, device=h_anchor.device)
    return F.cross_entropy(logits, labels)


def link_regularization_loss(p_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    L_reg -- binary cross-entropy standard sur l'existence/intensité du lien.

    Args:
        p_pred: (E,) -- sorties du LinkPredictor
        y_true: (E,) -- labels 0/1 (ou intensité continue si pondéré)
    """
    #return F.binary_cross_entropy(p_pred, y_true)
    return F.mse_loss(p_pred, y_true)


def contrastive_alignment_loss(h_word: torch.Tensor, h_audio_pos: torch.Tensor, h_audio_all: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    L_contrast -- alignement acoustique-linguistique.

    IMPORTANT : h_audio_all doit exclure les arêtes masquées par le dropout
    de modalité ET par le masquage anti-fuite de supervision
    (cf. graph/dropout.py) -- ce sont deux mécanismes indépendants.

    Args:
        h_word: (P,) x D -- embeddings des nœuds texte ancres
        h_audio_pos: (P, D) -- embedding audio positif correspondant
        h_audio_all: (E, D) -- tous les embeddings audio disponibles au
            dénominateur (paires (i,j) in E, cf. formule des slides)
    """
    pos_sim = (h_word * h_audio_pos).sum(-1) / tau
    all_sim = h_word @ h_audio_all.T / tau  # (P, E)
    log_prob = pos_sim - torch.logsumexp(all_sim, dim=-1)
    return -log_prob.mean()


def total_loss(
    l_acoustic: torch.Tensor,
    l_reg: torch.Tensor,
    l_contrast: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Combine les 3 pertes -- alpha, beta fixés pour CETTE config d'entraînement."""
    return alpha * l_acoustic + beta * l_contrast + l_reg 
