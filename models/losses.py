"""
Les trois pertes d'entraînement - Version supervisée.
"""

import torch
import torch.nn.functional as F


def acoustic_loss(
    h_anchor: torch.Tensor, 
    h_positive: torch.Tensor, 
    h_negatives: torch.Tensor, 
    tau: float = 0.1
) -> torch.Tensor:
    """
    L_acoustic -- similarité locale intra-modale (audio-audio uniquement).
    Version supervisée : chaque ancre a autant de négatifs que de positifs.
    
    Args:
        h_anchor: (K, D) -- embeddings ancres
        h_positive: (K, D) -- embeddings positifs correspondants
        h_negatives: (K, M, D) -- M négatifs par paire (M = nombre de positifs par ancre)
        tau: température
    """
    pos_sim = (h_anchor * h_positive).sum(-1) / tau  # (K,)
    neg_sim = torch.einsum("kd,kmd->km", h_anchor, h_negatives) / tau  # (K, M)
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (K, 1+M)
    labels = torch.zeros(h_anchor.size(0), dtype=torch.long, device=h_anchor.device)
    return F.cross_entropy(logits, labels)


def contrastive_alignment_loss(
    h_word: torch.Tensor, 
    h_audio_pos: torch.Tensor, 
    h_audio_neg: torch.Tensor, 
    tau: float = 0.1
) -> torch.Tensor:
    """
    L_contrast -- alignement acoustique-linguistique (version supervisée).
    Chaque mot a autant de négatifs audio que de positifs.
    
    Args:
        h_word: (K, D) -- embeddings des mots ancres
        h_audio_pos: (K, D) -- embeddings audio positifs correspondants
        h_audio_neg: (K, M, D) -- M négatifs audio par paire (M = nombre de positifs par mot)
        tau: température
    """
    pos_sim = (h_word * h_audio_pos).sum(-1) / tau  # (K,)
    neg_sim = torch.einsum("kd,kmd->km", h_word, h_audio_neg) / tau  # (K, M)
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (K, 1+M)
    labels = torch.zeros(h_word.size(0), dtype=torch.long, device=h_word.device)
    return F.cross_entropy(logits, labels)


def link_regularization_loss(p_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """L_reg -- binary cross-entropy standard."""
    #return F.mse_loss(p_pred, y_true)
    return F.binary_cross_entropy(p_pred, y_true)


def total_loss(
    l_acoustic: torch.Tensor,
    l_reg: torch.Tensor,
    l_contrast: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Combine les 3 pertes."""
    return alpha * l_acoustic + beta * l_contrast + l_reg
