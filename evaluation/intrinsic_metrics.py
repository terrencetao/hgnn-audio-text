"""
Métriques intrinsèques -- Alignment & Uniformity (Wang & Isola, ICML 2020).

Référence : https://arxiv.org/abs/2005.10242
Implémentation officielle : https://github.com/SsnL/align_uniform

Utilisées comme Étage 1 du protocole de validation croisée
(training/cross_validation.py) -- TOUJOURS ensemble, jamais l'une sans
l'autre (cf. discussion : l'alignment seul est trivialement optimal en
cas de collapse total).
"""

import torch


def alignment(h_positive_a: torch.Tensor, h_positive_b: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    L_align = E[ ||f(x) - f(x+)||^alpha ]

    Args:
        h_positive_a, h_positive_b: (N_pairs, D) -- embeddings normalisés
            des deux côtés de chaque paire positive (audio, texte apparié)
    """
    return (h_positive_a - h_positive_b).norm(dim=-1).pow(alpha).mean()


def uniformity(h_all: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    L_uniform = log E[ exp(-t ||f(x) - f(y)||^2) ]

    Args:
        h_all: (N, D) -- tous les embeddings normalisés (mélange audio+texte
            ou par modalité séparément, selon ce qu'on veut diagnostiquer)
    """
    sq_dists = torch.pdist(h_all, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()


def alignment_uniformity_score(
    h_positive_a: torch.Tensor,
    h_positive_b: torch.Tensor,
    h_all: torch.Tensor,
) -> dict[str, float]:
    """
    Calcule les deux métriques conjointement -- point d'entrée à utiliser
    dans la boucle de validation croisée (jamais l'une des deux isolément).
    """
    align = alignment(h_positive_a, h_positive_b).item()
    unif = uniformity(h_all).item()
    return {"alignment": align, "uniformity": unif}
