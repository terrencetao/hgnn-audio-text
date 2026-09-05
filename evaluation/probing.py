"""
Batterie de probing linguistique -- diagnostics POST-HOC uniquement,
JAMAIS utilisés dans training/cross_validation.py (cf. README, principe
d'organisation).

Références méthodologiques :
- Pasad, Chou & Livescu, "Layer-wise Analysis of a Self-supervised Speech
  Representation Model", ASRU 2021.
- Conneau et al., "What You Can Cram Into a Single Vector: Probing
  Sentence Embeddings for Linguistic Properties", ACL 2018.

Rappel des décisions prises dans la discussion :
- Ces probes s'appliquent au niveau PHRASE (vecteur pooled), pas mot --
  la classification tonale et la classification fermée de mots proposées
  initialement ne s'appliquent plus telles quelles (cf. discussion
  "on est passé à la représentation de phrase").
- Le probing locuteur n'est PAS un score à maximiser : un score proche du
  hasard est un signe POSITIF (bonne purge du signal non-sémantique par
  L_contrast). Voir la fonction `interpret_speaker_probing`.
- La longueur de phrase sert de tâche TÉMOIN (contrôle négatif) : si le
  probe la prédit anormalement bien, c'est un signal d'alerte
  méthodologique sur les autres probes, pas un résultat à interpréter.
"""

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import torch


@dataclass
class ProbingResult:
    task_name: str
    accuracy: float
    chance_level: float
    interpretation: str


def linear_probe(embeddings: np.ndarray, labels: np.ndarray, n_folds: int = 5) -> float:
    """Classifieur linéaire simple sur embeddings gelés -- brique commune à tous les probes."""
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, embeddings, labels, cv=n_folds)
    return float(scores.mean())


def probe_speaker_identity(embeddings: np.ndarray, speaker_ids: np.ndarray) -> ProbingResult:
    """
    cf. discussion "les données linguistiques ne peuvent pas aider à la
    reconnaissance du locuteur" -- un score BAS est le résultat attendu et
    souhaitable si L_contrast domine bien la loss combinée.
    """
    acc = linear_probe(embeddings, speaker_ids)
    chance = 1.0 / len(np.unique(speaker_ids))
    interpretation = (
        "Score proche du hasard attendu et souhaitable (purge du signal "
        "non-sémantique par L_contrast). Un score élevé signale que "
        "L_acoustic domine trop (alpha trop grand) -- à recouper avec les "
        "hyperparamètres retenus en cross-validation."
    )
    return ProbingResult("speaker_identity", acc, chance, interpretation)


def probe_sentence_length(embeddings: np.ndarray, lengths: np.ndarray, n_bins: int = 4) -> ProbingResult:
    """
    TÂCHE TÉMOIN (contrôle négatif) -- cf. Conneau et al. 2018.
    Si ce probe obtient un score anormalement élevé, c'est un signal
    d'ALERTE méthodologique, pas un résultat positif à interpréter comme tel.
    """
    length_bins = np.digitize(lengths, np.percentile(lengths, [25, 50, 75]))
    acc = linear_probe(embeddings, length_bins)
    chance = 1.0 / n_bins
    interpretation = (
        "TÂCHE TÉMOIN : un score élevé ici est un signal d'alerte "
        "(le probe capture peut-être un artefact superficiel), pas un "
        "résultat à interpréter positivement pour lui-même."
    )
    return ProbingResult("sentence_length_control", acc, chance, interpretation)


def probe_tonal_contrast_presence(embeddings: np.ndarray, has_tonal_contrast: np.ndarray) -> ProbingResult:
    """
    Version phrase (affaiblie) de la classification tonale -- prédit la
    PRÉSENCE d'un contraste tonal dans la phrase, pas "le" ton (qui n'a
    plus de sens univoque au niveau phrase). Spécifique au Yemba.
    """
    acc = linear_probe(embeddings, has_tonal_contrast)
    chance = 0.5
    interpretation = (
        "Ici, contrairement au locuteur, un score ÉLEVÉ est souhaitable "
        "sans ambiguïté : le ton porte du sens en Yemba, il doit être "
        "préservé par la représentation."
    )
    return ProbingResult("tonal_contrast_presence", acc, chance, interpretation)


def abx_test(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    embeddings_x: torch.Tensor,
    x_closer_to_a: torch.Tensor,
) -> float:
    """
    Test ABX (ZeroSpeech) -- AUCUN entraînement de classifieur nécessaire,
    juste une comparaison de distances. Le plus léger de tous les probes,
    cohérent avec l'esprit "frugal" du projet.

    Args:
        embeddings_a, embeddings_b, embeddings_x: (N, D) -- triplets
        x_closer_to_a: (N,) booléen -- vérité terrain (X est censé être
            plus proche de A que de B selon une référence linguistique connue)

    Returns:
        accuracy : proportion de triplets où d(X,A) < d(X,B) correspond
            bien à x_closer_to_a
    """
    dist_xa = (embeddings_x - embeddings_a).norm(dim=-1)
    dist_xb = (embeddings_x - embeddings_b).norm(dim=-1)
    predicted_closer_to_a = dist_xa < dist_xb
    return (predicted_closer_to_a == x_closer_to_a).float().mean().item()


def run_probing_battery(embeddings: np.ndarray, metadata: dict) -> list[ProbingResult]:
    """
    Point d'entrée principal -- lance tous les probes disponibles selon
    les métadonnées fournies. Toujours exécuter EN DERNIER, après
    entraînement final (jamais pendant la cross-validation).
    """
    results = [probe_sentence_length(embeddings, metadata["lengths"])]  # témoin en premier
    if "speaker_ids" in metadata:
        results.append(probe_speaker_identity(embeddings, metadata["speaker_ids"]))
    if "has_tonal_contrast" in metadata:
        results.append(probe_tonal_contrast_presence(embeddings, metadata["has_tonal_contrast"]))
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    # TODO: charger le modèle + checkpoint, extraire les embeddings de
    # validation/test, puis appeler run_probing_battery(...)
    print(f"Probing battery -- config={args.config}, checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()

