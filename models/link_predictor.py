"""
Link predictor -- Linear-ReLU-Linear-Sigmoid (cf. slide "Méthode frugale
basée sur le GNN").

Double rôle établi dans la discussion :
1. Pendant l'entraînement : calcule L_reg (régularisation, existence/intensité
   d'un lien).
2. À l'inférence (Option 3) : composant ACTIF, propose de nouvelles arêtes
   pour un nœud orphelin plutôt que d'évaluer des arêtes déjà connues.
"""

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_audio: torch.Tensor, h_word: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_audio: (B, D)
            h_word: (B, D) -- déjà apparié 1-à-1 avec h_audio (paires à scorer)
        Returns:
            p: (B,) -- probabilité de lien, in [0, 1]
        """
        pair = torch.cat([h_audio, h_word], dim=-1)
        return self.net(pair).squeeze(-1)

    def propose_edges(
        self, h_audio_query: torch.Tensor, h_word_candidates: torch.Tensor, top_k: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Utilisé UNIQUEMENT à l'inférence (Option 3) : étant donné un
        embedding audio orphelin (déjà calculé via l'Option 2, ancré dans
        le graphe hétérogène de référence), propose les top_k nœuds texte
        candidats les plus probables.

        Args:
            h_audio_query: (D,) -- embedding d'un seul nœud orphelin
            h_word_candidates: (N_words, D) -- tous les candidats texte disponibles

        Returns:
            top_k_indices: (effective_top_k,)
            top_k_scores: (effective_top_k,) -- probabilités, pour le seuillage de confiance

        CORRECTION (cf. discussion) : `top_k` est maintenant clampé à
        `h_word_candidates.size(0)`. Sans ça, un appel avec top_k=100 sur
        un graphe de référence à moins de 100 mots (ex: Yemba -- 60 mots
        au total, souvent moins après filtrage par reference_fraction)
        lève `RuntimeError: selected index k out of range`. Le clamp est
        fait ICI (pas au point d'appel dans inference_options.py) pour
        protéger tous les appelants actuels et futurs, pas seulement
        Option 3.
        """
        num_candidates = h_word_candidates.size(0)
        if num_candidates == 0:
            empty = torch.empty(0, dtype=torch.long, device=h_word_candidates.device)
            return empty, torch.empty(0, device=h_word_candidates.device)

        effective_top_k = min(top_k, num_candidates)  # FIX -- clamp

        h_query_expanded = h_audio_query.unsqueeze(0).expand(num_candidates, -1)
        scores = self.forward(h_query_expanded, h_word_candidates)  # (N_words,)
        top_k_scores, top_k_indices = scores.topk(effective_top_k)
        return top_k_indices, top_k_scores
