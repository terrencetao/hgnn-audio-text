"""
Sélection d'hyperparamètres par validation croisée -- protocole à 2 étages.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, Any, List, Tuple, Callable
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import yaml
from pathlib import Path


@dataclass
class HyperparamConfig:
    """Configuration d'hyperparamètres à tester."""
    alpha: float
    beta: float
    orphan_fraction: float
    tau: float = 0.1
    gnn_layers: int = 2
    lr: float = 0.001
    dropout: float = 0.1
    weight_decay: float = 1e-5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'orphan_fraction': self.orphan_fraction,
            'tau': self.tau,
            'gnn_layers': self.gnn_layers,
            'lr': self.lr,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay
        }
    
    def to_id(self, index: int) -> str:
        """Génère un ID unique pour cette configuration."""
        return f"config_{index:04d}"


def generate_grid(cv_config: dict) -> list[HyperparamConfig]:
    """Construit la grille de configs."""
    alpha_grid = cv_config.get('alpha_grid', [0.3, 0.5, 0.7])
    beta_grid = cv_config.get('beta_grid', [0.3, 0.5, 0.7])
    orphan_grid = cv_config.get('orphan_fraction_grid', [0.2, 0.3, 0.4])
    tau_grid = cv_config.get('tau_grid', [0.01, 0.05, 0.1, 0.2])
    layers_grid = cv_config.get('gnn_layers_grid', [2])
    lr_grid = cv_config.get('lr_grid', [0.001])
    dropout_grid = cv_config.get('dropout_grid', [0.1])
    weight_decay_grid = cv_config.get('weight_decay_grid', [1e-5])
    
    combos = product(
        alpha_grid,
        beta_grid,
        orphan_grid,
        tau_grid,
        layers_grid,
        lr_grid,
        dropout_grid,
        weight_decay_grid
    )
    
    configs = []
    for alpha, beta, orphan, tau, layers, lr, dropout, weight_decay in combos:
        configs.append(HyperparamConfig(
            alpha=alpha,
            beta=beta,
            orphan_fraction=orphan,
            tau=tau,
            gnn_layers=layers,
            lr=lr,
            dropout=dropout,
            weight_decay=weight_decay
        ))
    
    print(f"📊 Grille générée: {len(configs)} configurations")
    return configs


def stage1_filter_candidates(
    results_by_config: dict[str, dict],
    top_k: int = 5,
) -> list[str]:
    """
    Étage 1 : sélectionne les configurations du front de Pareto.
    """
    if not results_by_config:
        return []
    
    config_ids = list(results_by_config.keys())
    
    # Extraire les métriques
    metrics = []
    for cid in config_ids:
        metrics.append({
            'id': cid,
            'alignment': results_by_config[cid]["alignment"],
            'uniformity': results_by_config[cid]["uniformity"]
        })
    
    # 🔑 Trouver le front de Pareto
    pareto_front = _find_pareto_front(metrics)
    
    # Si le front est vide ou très petit, utiliser le top_k par rang
    if len(pareto_front) == 0:
        print(f"   ⚠️ Front de Pareto vide ou trop petit, utilisation du rang")
        return _rank_by_combined_score(metrics, top_k)
    
    print(f"   Front de Pareto: {len(pareto_front)} configurations")
    
    # Si le front est plus grand que top_k, réduire
    if len(pareto_front) > top_k:
        # Utiliser le rang pour réduire
        sorted_front = sorted(
            pareto_front,
            key=lambda x: x['alignment'] + x['uniformity']
        )
        pareto_front = sorted_front[:top_k]
        print(f"   Réduit à {len(pareto_front)} configurations (top {top_k})")
    
    return [m['id'] for m in pareto_front]


def _rank_by_combined_score(metrics: list[dict], top_k: int) -> list[str]:
    """
    Alternative : classe par score combiné (rang Borda).
    """
    # Extraire les valeurs
    align_values = [m['alignment'] for m in metrics]
    unif_values = [m['uniformity'] for m in metrics]
    
    align_ranks = _rank(align_values)
    unif_ranks = _rank(unif_values)
    
    combined_rank = [a + u for a, u in zip(align_ranks, unif_ranks)]
    
    # Trier par rang combiné
    sorted_metrics = sorted(
        zip(metrics, combined_rank),
        key=lambda x: x[1]
    )
    
    return [m[0]['id'] for m in sorted_metrics[:top_k]]


def _find_pareto_front(metrics: list[dict]) -> list[dict]:
    """
    Trouve le front de Pareto pour minimisation des deux objectifs.
    """
    pareto_front = []
    
    for i, point in enumerate(metrics):
        is_dominated = False
        
        for j, other in enumerate(metrics):
            if i == j:
                continue
            
            # Vérifier si other domine point
            if (other['alignment'] <= point['alignment'] and 
                other['uniformity'] <= point['uniformity'] and
                (other['alignment'] < point['alignment'] or 
                 other['uniformity'] < point['uniformity'])):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(point)
    
    return pareto_front


def stage2_select_best(
    candidate_ids: list[str],  # 🔑 IDs des candidats
    link_predictor_outputs: dict[str, tuple],  # 🔑 clé = config_id
    config_by_id: dict[str, HyperparamConfig],  # 🔑 Mapping ID → config
) -> HyperparamConfig:
    """
    Étage 2 : parmi les candidats, retient celui avec la meilleure MSE.
    """
    if not candidate_ids:
        return None
    
    best_config, best_mse = None, float('inf')
    
    for config_id in candidate_ids:
        p_pred, y_true = link_predictor_outputs[config_id]
        
        if torch.is_tensor(p_pred):
            p_pred = p_pred.cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        
        p_pred = np.clip(p_pred, 0, 1)
        
        try:
            mse = mean_squared_error(y_true, p_pred)
            if mse < best_mse:
                best_mse = mse
                best_config = config_by_id[config_id]
        except Exception as e:
            print(f"   ⚠️ Erreur MSE pour {config_id}: {e}")
            continue
    
    if best_config is not None:
        print(f"   ✅ Meilleur MSE: {best_mse:.6f}")
    
    return best_config


def _rank(values: list[float]) -> list[int]:
    """Rang 0 = valeur la plus basse."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def run_cross_validation(
    cv_config: dict,
    train_and_eval_fn: Callable,
    base_config_path: str = "config/default.yaml"
) -> HyperparamConfig:
    """Boucle principale de validation croisée."""
    print("="*60)
    print("🔬 VALIDATION CROISÉE")
    print("="*60)
    
    grid = generate_grid(cv_config)
    print(f"   {len(grid)} configurations à tester")
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 🔑 Dictionnaires avec IDs comme clés
    results_by_config = {}  # config_id → {'alignment': ..., 'uniformity': ...}
    link_predictor_outputs = {}  # config_id → (p_pred, y_true)
    config_by_id = {}  # config_id → HyperparamConfig
    
    # Tester chaque configuration
    for i, config in enumerate(grid):
        config_id = config.to_id(i + 1)
        config_by_id[config_id] = config
        
        print(f"\n🔍 Test {i+1}/{len(grid)}:")
        print(f"   ID: {config_id}")
        print(f"   alpha={config.alpha}, beta={config.beta}, orphan={config.orphan_fraction}")
        print(f"   tau={config.tau}, layers={config.gnn_layers}, lr={config.lr}, dropout={config.dropout}")
        
        try:
            result = train_and_eval_fn(config)
            
            results_by_config[config_id] = {
                "alignment": result["alignment"],
                "uniformity": result["uniformity"],
            }
            link_predictor_outputs[config_id] = result["link_pred_outputs"]
            
            print(f"   ✅ Alignment={result['alignment']:.4f}, Uniformity={result['uniformity']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results_by_config:
        raise RuntimeError("Aucune configuration n'a réussi !")
    
    # Étape 1: Filtrer les candidats
    print(f"\n📊 Étape 1: Filtrage des candidats")
    candidates_ids = stage1_filter_candidates(results_by_config, top_k=cv_config.get('top_k', 5))
    print(f"   {len(candidates_ids)} candidats retenus")
    
    # Étape 2: Sélectionner le meilleur avec MSE
    print(f"\n🎯 Étape 2: Sélection du meilleur (MSE)")
    best_config = stage2_select_best(candidates_ids, link_predictor_outputs, config_by_id)
    
    if best_config is None:
        raise RuntimeError("Aucun candidat valide pour l'étape 2")
    
    print(f"\n✅ Meilleure configuration trouvée:")
    print(f"   - alpha: {best_config.alpha}")
    print(f"   - beta: {best_config.beta}")
    print(f"   - orphan_fraction: {best_config.orphan_fraction}")
    print(f"   - tau: {best_config.tau}")
    print(f"   - gnn_layers: {best_config.gnn_layers}")
    print(f"   - lr: {best_config.lr}")
    print(f"   - dropout: {best_config.dropout}")
    print(f"   - weight_decay: {best_config.weight_decay}")
    
    return best_config


def create_cv_folds(
    n_examples: int,
    n_folds: int = 5,
    by_speaker: bool = False,
    speaker_labels: List[str] = None,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Crée des folds pour la validation croisée."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    if by_speaker and speaker_labels is not None:
        from collections import defaultdict
        speaker_indices = defaultdict(list)
        for idx, speaker in enumerate(speaker_labels):
            speaker_indices[speaker].append(idx)
        
        speakers = list(speaker_indices.keys())
        random.shuffle(speakers)
        
        n_speakers_per_fold = len(speakers) // n_folds
        folds = []
        
        for i in range(n_folds):
            start = i * n_speakers_per_fold
            end = (i + 1) * n_speakers_per_fold if i < n_folds - 1 else len(speakers)
            val_speakers = speakers[start:end]
            train_speakers = [s for s in speakers if s not in val_speakers]
            
            train_indices = []
            for speaker in train_speakers:
                train_indices.extend(speaker_indices[speaker])
            
            val_indices = []
            for speaker in val_speakers:
                val_indices.extend(speaker_indices[speaker])
            
            folds.append((train_indices, val_indices))
        
        print(f"✅ {n_folds} folds créés (par locuteur)")
        for i, (train, val) in enumerate(folds):
            print(f"   Fold {i+1}: Train={len(train)}, Val={len(val)}")
        
        return folds
    
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        indices = list(range(n_examples))
        return [(train.tolist(), val.tolist()) for train, val in kf.split(indices)]
