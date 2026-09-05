#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/yemba.yaml}
CHECKPOINT=${2:-runs/yemba_hgnn/best.ckpt}

# Évaluations POST-HOC uniquement (jamais utilisées pour sélectionner les
# hyperparamètres, cf. README) : retrieval, batterie de probing, options
# d'inférence 1/2/3.
python -m evaluation.retrieval --config "$CONFIG" --checkpoint "$CHECKPOINT"
python -m evaluation.probing --config "$CONFIG" --checkpoint "$CHECKPOINT"
