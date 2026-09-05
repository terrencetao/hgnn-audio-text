# HGNN Audio-Texte — Alignement frugal d'espaces pour langues faiblement dotées

Squelette de projet organisé pour refléter directement les décisions prises
lors de la conception (voir `presentation.pdf` du projet associé).

## Organisation

```
hgnn-audio-text/
├── config/                  # Un fichier YAML par expérience (dataset + hyperparamètres)
│   ├── timit.yaml
│   └── yemba.yaml
├── data/                    # Chargement et préparation des données brutes
│   ├── datasets.py          # Dataset PyTorch : paires (audio, texte, id_locuteur, ...)
│   ├── prepare_timit.py
│   └── prepare_yemba.py
├── graph/                   # Construction et manipulation du graphe hétérogène
│   ├── build_graph.py       # Assemble G_a, G_l, arêtes croisées (G_a <-> G_l)
│   ├── similarity.py        # Similarité linguistique (LaBSE / sac de phonèmes / articulatoire)
│   └── dropout.py           # Dropout de modalité SOFT (masquage local, re-tiré à chaque itération)
├── models/                  # Modules du réseau
│   ├── encoders.py          # Wrapper XLSR/wav2vec2 + AttentivePooling (mot -> phrase)
│   ├── gnn.py                # Encodeur GraphSAGE hétérogène
│   ├── link_predictor.py    # Linear-ReLU-Linear-Sigmoid, réutilisé aussi à l'inférence (Option 3)
│   └── losses.py             # L_acoustic, L_reg, L_contrast
├── training/                 # Boucle d'entraînement et sélection d'hyperparamètres
│   ├── train.py               # Boucle principale (une config donnée)
│   └── cross_validation.py   # Protocole à 2 étages : (Alignment+Uniformity) -> AUC link predictor
├── evaluation/                # Tout ce qui ne sert PAS à sélectionner les hyperparamètres
│   ├── intrinsic_metrics.py  # Alignment, Uniformity (réutilisées aussi en CV)
│   ├── retrieval.py          # WO@k / PO@k, Recall@k (mono- et cross-lingue)
│   ├── probing.py             # Batterie de probing (locuteur, ton, longueur = contrôle négatif, ABX)
│   └── inference_options.py  # Option 2 (ancrage hétérogène) / Option 3 (+ reconnexion dynamique)
├── scripts/                   # Points d'entrée CLI
│   ├── run_training.sh
│   └── run_eval.sh
└── notebooks/                  # Exploration ponctuelle, jamais de logique critique dedans
```

## Principe d'organisation

- **`graph/` est isolé de `models/`** : la construction du graphe (structure, arêtes, dropout)
  ne dépend d'aucun poids appris — elle peut être testée et débogée indépendamment du réseau.
- **`training/cross_validation.py` n'importe QUE des fonctions de `evaluation/intrinsic_metrics.py`
  et `models/link_predictor.py`** — jamais `evaluation/retrieval.py` ni `probing.py`. C'est la
  décision qu'on a prise : la sélection d'hyperparamètres reste agnostique à la tâche finale.
- **`evaluation/inference_options.py` ne réentraîne jamais rien** : il prend un modèle déjà
  entraîné (checkpoint) et un graphe de référence, et implémente uniquement les Options 2/3
  (l'Option 1 est disponible comme ablation, cf. `--ablation-audio-only` dans les scripts).

## Ordre d'implémentation recommandé

1. `graph/build_graph.py` + `graph/similarity.py` — sans ça, rien d'autre n'est testable
2. `models/encoders.py` (AttentivePooling) + `models/gnn.py` — le forward pass de base
3. `models/losses.py` — les 3 pertes
4. `graph/dropout.py` — brancher le dropout soft dans la boucle d'entraînement
5. `training/train.py` — boucle complète pour UNE config fixée manuellement
6. `evaluation/intrinsic_metrics.py` + `training/cross_validation.py` — une fois qu'un
   entraînement complet fonctionne, automatiser la sélection d'hyperparamètres
7. `evaluation/retrieval.py`, `probing.py`, `inference_options.py` — en dernier, une fois
   qu'un modèle "final" existe

## Dépendances principales

Voir `requirements.txt`. Stack retenue : PyTorch + PyTorch Geometric (GraphSAGE hétérogène
via `HeteroConv`/`SAGEConv`), `transformers` (encodeur wav2vec2/XLSR), `scikit-learn`
(silhouette score, classifieurs de probing).
