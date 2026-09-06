"""
run_retrieval_experiments.py

Script pour exécuter des expériences de retrieval avec différentes combinaisons de paramètres.

Le seuil de similarité est FIXÉ par la config (config.graph_build.similarity_threshold)
et n'est PAS un hyperparamètre à explorer.

Paramètres explorés:
- reference_fraction: Fraction du graphe à utiliser comme référence
- max_edges_per_node: Nombre max d'arêtes par nouveau nœud
- k_values: Nombre de voisins pour le retrieval
- representation_type: Type de représentation linguistique

Ces expériences sont spécifiques à un backbone et un modèle linguistique donnés.

CORRECTIONS apportées (cf. discussion) par rapport à la version précédente :

1. `from retrieval import RetrievalEvaluator` référençait un module à la
   racine du projet, incohérent avec la structure réelle où le fichier vit
   dans `evaluation/retrieval.py`. Corrigé en `from evaluation.retrieval
   import RetrievalEvaluator`.

2. `best_configs` et `summary_df` étaient référencés dans la construction
   de `global_summary`, hors du bloc `if all_results:` où ils sont
   définis -- si TOUTES les expériences de la grille échouent (all_results
   vide), le script levait un NameError au lieu de sauvegarder un résumé
   d'échec informatif. Ajout d'une garde explicite.
"""

import sys
from pathlib import Path
# Ajouter le dossier parent au path (racine du projet, depuis scripts/)
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from typing import List, Dict, Any
import itertools

# CORRECTION point 1 : import path cohérent avec evaluation/retrieval.py
from evaluation.retrieval import RetrievalEvaluator
from graph.similarity import LinguisticRepresentation


def get_model_names(config_path: str) -> tuple:
    """Récupère les noms des modèles depuis la configuration."""
    config = OmegaConf.load(config_path)
    backbone_name = config['encoder']['backbone']
    backbone_model_name = backbone_name.split('/')[1] if '/' in backbone_name else backbone_name
    representation_name = config['linguistic']['representation']
    threshold = config.graph.build.similarity_threshold
    return backbone_model_name, representation_name, threshold


def run_single_experiment(
    config_path: str,
    checkpoint_path: str,
    test_csv_path: str,
    reference_fraction: float,
    max_edges_per_node: int,
    k_values: List[int],
    representation_type: str,
    seed: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Exécute une expérience unique avec les paramètres donnés.
    """
    rep_type = LinguisticRepresentation(representation_type)

    evaluator = RetrievalEvaluator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        representation_type=rep_type,
        max_edges_per_node=max_edges_per_node,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    results = evaluator.evaluate_on_test_set(
        reference_fraction=reference_fraction,
        test_csv_path=test_csv_path,
        k_values=k_values,
        seed=seed
    )

    backbone_name, rep_name, threshold = get_model_names(config_path)
    results["experiment_params"] = {
        "backbone_model": backbone_name,
        "representation_model": rep_name,
        "similarity_threshold": threshold,  # Fixé par la config
        "reference_fraction": reference_fraction,
        "max_edges_per_node": max_edges_per_node,
        "k_values": k_values,
        "representation_type": representation_type,
        "seed": seed,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "test_csv_path": test_csv_path,
        "timestamp": datetime.now().isoformat()
    }

    return results


def save_experiment_results(
    results: Dict[str, Any],
    output_dir: Path,
    experiment_name: str = None
) -> Path:
    """Sauvegarde les résultats d'une expérience."""
    output_dir = Path(output_dir) / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        params = results["experiment_params"]
        experiment_name = f"retrieval_{params['backbone_model']}_{params['representation_model']}_f{params['reference_fraction']}_thr{params['similarity_threshold']}_max{params['max_edges_per_node']}_seed{params['seed']}"

    pt_path = output_dir / f"{experiment_name}.pt"
    torch.save(results, pt_path)

    json_path = output_dir / f"{experiment_name}.json"
    json_data = {}
    for key, value in results.items():
        if key in ["metadata", "experiment_params"]:
            json_data[key] = value
        elif isinstance(value, dict):
            json_data[key] = {str(k): float(v) for k, v in value.items()}
        else:
            json_data[key] = value

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    return pt_path


def run_grid_experiment(
    config_path: str,
    checkpoint_path: str,
    test_csv_path: str,
    output_dir: Path,
    reference_fractions: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    max_edges_per_nodes: List[int] = [5, 10, 20],
    k_values: List[int] = [1, 5, 10],
    representation_types: List[str] = ["labse"],
    seeds: List[int] = [42],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exécute une grille d'expériences.

    Le seuil de similarité est fixé par la config et N'EST PAS exploré.
    """
    backbone_model, rep_model, threshold = get_model_names(config_path)

    print(f"\n🎯 EXPÉRIENCES POUR:")
    print(f"   - Backbone: {backbone_model}")
    print(f"   - Représentation linguistique: {rep_model}")
    print(f"   - Seuil de similarité (fixé): {threshold}")
    print("=" * 80)

    # Créer toutes les combinaisons (sans le seuil qui est fixé)
    param_combinations = list(itertools.product(
        reference_fractions,
        max_edges_per_nodes,
        representation_types,
        seeds
    ))

    all_results = {}
    failed_experiments = []

    print(f"\n🚀 Lancement de {len(param_combinations)} expériences...")

    for i, (ref_frac, max_edges, rep_type, seed) in enumerate(tqdm(param_combinations, desc="Expériences")):
        try:
            exp_name = f"retrieval_{backbone_model}_{rep_model}_f{ref_frac}_thr{threshold}_max{max_edges}_{rep_type}_seed{seed}"

            results = run_single_experiment(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                test_csv_path=test_csv_path,
                reference_fraction=ref_frac,
                max_edges_per_node=max_edges,
                k_values=k_values,
                representation_type=rep_type,
                seed=seed,
                output_dir=output_dir
            )

            save_experiment_results(results, output_dir, exp_name)
            all_results[exp_name] = results

            if verbose:
                recall1 = results.get("option2_reference", {}).get(1, 0.0)
                print(f"✅ {i+1}/{len(param_combinations)}: {exp_name} -> R@1={recall1:.4f}")

        except Exception as e:
            failed_experiments.append({
                "params": {
                    "reference_fraction": ref_frac,
                    "max_edges_per_node": max_edges,
                    "representation_type": rep_type,
                    "seed": seed
                },
                "error": str(e)
            })
            print(f"❌ Échec: {str(e)}")
            continue

    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DES EXPÉRIENCES")
    print("=" * 80)
    print(f"Backbone: {backbone_model}")
    print(f"Représentation: {rep_model}")
    print(f"Seuil: {threshold}")
    print(f"Total: {len(param_combinations)} expériences")
    print(f"Réussies: {len(all_results)}")
    print(f"Échouées: {len(failed_experiments)}")

    # CORRECTION point 2 : summary_df/best_configs initialisés AVANT le
    # bloc conditionnel, pour être toujours définis (même vides) au
    # moment de construire global_summary plus bas -- évite le NameError
    # si all_results est vide (toutes les expériences ont échoué).
    summary_df = pd.DataFrame()
    best_configs = pd.DataFrame()

    if all_results:
        summary_data = []
        for exp_name, results in all_results.items():
            params = results["experiment_params"]
            summary_data.append({
                "experiment": exp_name,
                "backbone": params["backbone_model"],
                "representation": params["representation_model"],
                "similarity_threshold": params["similarity_threshold"],
                "reference_fraction": params["reference_fraction"],
                "max_edges_per_node": params["max_edges_per_node"],
                "R@1_option1": results.get("option1_ablation", {}).get(1, 0.0),
                "R@1_option2": results.get("option2_reference", {}).get(1, 0.0),
                "R@1_option3": results.get("option3_reconnection", {}).get(1, 0.0),
                "R@5_option2": results.get("option2_reference", {}).get(5, 0.0),
                "R@10_option2": results.get("option2_reference", {}).get(10, 0.0),
                "avg_edges": results.get("metadata", {}).get("avg_edges_per_query", 0),
                "num_queries": results.get("metadata", {}).get("num_test_samples", 0)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "retrieval" / f"experiments_summary_{backbone_model}_{rep_model}_thr{threshold}.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n🏆 MEILLEURES CONFIGURATIONS (R@1 sur Option 2):")
        print("-" * 80)
        best_configs = summary_df.nlargest(5, "R@1_option2")
        for _, row in best_configs.iterrows():
            print(f"  R@1={row['R@1_option2']:.4f} | ref_frac={row['reference_fraction']:.2f} | "
                  f"max_edges={row['max_edges_per_node']}")

        print("\n📈 STATISTIQUES:")
        print("-" * 80)
        print(f"  R@1 Option 2 - Moyenne: {summary_df['R@1_option2'].mean():.4f} ± {summary_df['R@1_option2'].std():.4f}")
        print(f"  R@1 Option 2 - Min/Max: {summary_df['R@1_option2'].min():.4f} / {summary_df['R@1_option2'].max():.4f}")
    else:
        print("\n⚠️  Aucune expérience réussie -- pas de résumé statistique à produire.")

    global_summary = {
        "backbone_model": backbone_model,
        "representation_model": rep_model,
        "similarity_threshold": threshold,
        "total_experiments": len(param_combinations),
        "successful": len(all_results),
        "failed": len(failed_experiments),
        # CORRECTION point 2 : garde explicite, plus de NameError possible
        "best_config": best_configs.iloc[0].to_dict() if not summary_df.empty and not best_configs.empty else None,
        "timestamp": datetime.now().isoformat(),
        "failed_experiments": failed_experiments
    }

    output_summary_dir = output_dir / "retrieval"
    output_summary_dir.mkdir(parents=True, exist_ok=True)
    with open(output_summary_dir / f"global_summary_{backbone_model}_{rep_model}_thr{threshold}.json", 'w') as f:
        json.dump(global_summary, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Exécute des expériences de retrieval"
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")

    # Paramètres à explorer
    parser.add_argument("--reference_fractions", type=float, nargs="+",
                       default=[0.1, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--max_edges_per_nodes", type=int, nargs="+",
                       default=[5, 10, 20])
    parser.add_argument("--k_values", type=int, nargs="+",
                       default=[1, 5, 10])
    parser.add_argument("--representation_types", type=str, nargs="+",
                       default=["labse"],
                       choices=["labse", "bag_of_phonemes", "articulatory"])
    parser.add_argument("--seeds", type=int, nargs="+",
                       default=[42])

    # Mode single
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--reference_fraction", type=float, default=0.5)
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    parser.add_argument("--representation_type", type=str, default="labse",
                       choices=["labse", "bag_of_phonemes", "articulatory"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_model, rep_model, threshold = get_model_names(args.config)

    print("=" * 80)
    print("🔍 RETRIEVAL EXPERIMENTS")
    print("=" * 80)
    print(f"Backbone: {backbone_model}")
    print(f"Représentation linguistique: {rep_model}")
    print(f"Seuil de similarité (config): {threshold}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    if args.single:
        print("\n🎯 Exécution d'une expérience unique")
        print("-" * 40)
        print(f"  reference_fraction: {args.reference_fraction}")
        print(f"  max_edges_per_node: {args.max_edges_per_node}")
        print(f"  k_values: {args.k_values}")
        print(f"  representation_type: {args.representation_type}")
        print(f"  seed: {args.seed}")
        print(f"  similarity_threshold (fixé): {threshold}")

        rep_type = LinguisticRepresentation(args.representation_type)

        evaluator = RetrievalEvaluator(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            representation_type=rep_type,
            max_edges_per_node=args.max_edges_per_node
        )

        results = evaluator.evaluate_on_test_set(
            reference_fraction=args.reference_fraction,
            test_csv_path=args.test_csv,
            k_values=args.k_values,
            seed=args.seed
        )

        exp_name = f"retrieval_{backbone_model}_{rep_model}_f{args.reference_fraction}_thr{threshold}_max{args.max_edges_per_node}_seed{args.seed}"
        save_experiment_results(results, output_dir, exp_name)

        print("\n📊 Résultats:")
        print("-" * 40)
        for option, scores in results.items():
            if option in ["metadata", "experiment_params"]:
                continue
            print(f"\n{option}:")
            for k, score in scores.items():
                print(f"  Recall@{k}: {score:.4f}")

        print(f"\n✅ Résultats sauvegardés dans {output_dir / 'retrieval'}")

    else:
        run_grid_experiment(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            test_csv_path=args.test_csv,
            output_dir=output_dir,
            reference_fractions=args.reference_fractions,
            max_edges_per_nodes=args.max_edges_per_nodes,
            k_values=args.k_values,
            representation_types=args.representation_types,
            seeds=args.seeds,
            verbose=True
        )

    print(f"\n✅ Toutes les expériences sont terminées!")
    print(f"📁 Résultats dans: {output_dir / 'retrieval'}")


if __name__ == "__main__":
    main()
