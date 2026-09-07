"""
run_cross_validation.py
Lance la validation croisée pour trouver les meilleurs hyperparamètres.
Sauvegarde toutes les métriques pour chaque combinaison.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import pickle
import argparse
import traceback
import csv
import json
from datetime import datetime
from collections import defaultdict

from cross_validation import (
    HyperparamConfig,
    run_cross_validation,
    create_cv_folds
)
from training.train import Trainer


# ============================================================================
# FONCTION PRINCIPALE DE LA CV
# ============================================================================

def train_and_evaluate_config(
    hyperparams: HyperparamConfig,
    base_config: dict,
    processed_dir: str = "./data/graphs/ewondo",
    output_dir: str = "./outputs/cv",
    fold: int = 0,
    train_indices: list = None,
    val_indices: list = None,
    epochs: int = 20,
    default_config_path: str = None,
) -> dict:
    """
    Entraîne et évalue une configuration pour la CV.
    
    Cette fonction crée un Trainer avec la configuration modifiée,
    l'entraîne, puis l'évalue.
    """
    try:
        # Construire la configuration modifiée
        config_override = {
            'loss_weights': {
                'alpha': hyperparams.alpha,
                'beta': hyperparams.beta
            },
            'dropout': {
                'orphan_fraction': hyperparams.orphan_fraction
            },
            'graph': {
                'gnn_layers': hyperparams.gnn_layers
            },
            'training': {
                'lr_gnn': hyperparams.lr,
                'dropout': hyperparams.dropout,
                'weight_decay': float(hyperparams.weight_decay),
                'epochs': epochs
            }
        }
        
        # Ajouter tau si présent
        if hasattr(hyperparams, 'tau') and hyperparams.tau is not None:
            config_override['training']['tau'] = hyperparams.tau
        
        # Créer le Trainer avec la configuration modifiée
        trainer = Trainer(
            config_path=default_config_path,
            config_override=config_override,
            processed_dir=processed_dir,
            output_dir=output_dir,
            train_indices=train_indices,
            val_indices=val_indices,
            fold=fold
        )
        
        # Entraîner
        train_results = trainer.train(epochs=epochs)
        
        # Évaluer (utilise les mêmes modèles que ceux entraînés)
        eval_results = trainer.evaluate()
        
        return {
            'alignment': eval_results['alignment'],
            'uniformity': eval_results['uniformity'],
            'link_pred_outputs': eval_results['link_pred_outputs'],
            'best_loss': train_results['best_loss'],
            'best_epoch': train_results['best_epoch']
        }
        
    except Exception as e:
        print(f"   ❌ Erreur dans train_and_evaluate_config: {e}")
        traceback.print_exc()
        return {
            'alignment': float('inf'),
            'uniformity': float('inf'),
            'link_pred_outputs': (torch.tensor([]), torch.tensor([])),
            'best_loss': float('inf'),
            'best_epoch': -1
        }


# ============================================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================================

def save_all_results(
    all_results: dict,
    output_dir: Path,
    best_config: HyperparamConfig,
    name_result: str,
):
    """
    Sauvegarde toutes les métriques pour chaque combinaison.
    
    Args:
        all_results: Dictionnaire avec toutes les métriques
        output_dir: Dossier de sortie
        best_config: Meilleure configuration trouvée
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sauvegarder en CSV
    csv_path = output_dir / f'all_results_{name_result}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'config_id',
            'alpha', 'beta', 'orphan_fraction', 'tau',
            'gnn_layers', 'lr', 'dropout', 'weight_decay',
            'avg_alignment', 'avg_uniformity',
            'best_loss', 'best_epoch',
            'num_folds_success', 'is_best'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for config_id, results in all_results.items():
            # Extraire les paramètres
            params = results['params']
            metrics = results['metrics']
            
            row = {
                'config_id': config_id,
                'alpha': params['alpha'],
                'beta': params['beta'],
                'orphan_fraction': params['orphan_fraction'],
                'tau': params.get('tau', 0.1),
                'gnn_layers': params['gnn_layers'],
                'lr': params['lr'],
                'dropout': params['dropout'],
                'weight_decay': params['weight_decay'],
                'avg_alignment': metrics['avg_alignment'],
                'avg_uniformity': metrics['avg_uniformity'],
                'best_loss': metrics['best_loss'],
                'best_epoch': metrics['best_epoch'],
                'num_folds_success': metrics['num_folds_success'],
                'is_best': 'YES' if results.get('is_best', False) else ''
            }
            writer.writerow(row)
    
    print(f"✅ Résultats CSV sauvegardés: {csv_path}")
    
    # 2. Sauvegarder en JSON (plus complet)
    json_path = output_dir / f'all_results_{name_result}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        # Convertir les tensors en listes pour JSON
        json_results = {}
        for config_id, results in all_results.items():
            json_results[config_id] = {
                'params': results['params'],
                'metrics': {
                    'avg_alignment': results['metrics']['avg_alignment'],
                    'avg_uniformity': results['metrics']['avg_uniformity'],
                    'best_loss': results['metrics']['best_loss'],
                    'best_epoch': results['metrics']['best_epoch'],
                    'num_folds_success': results['metrics']['num_folds_success']
                },
                'fold_results': results.get('fold_results', []),
                'is_best': results.get('is_best', False)
            }
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"✅ Résultats JSON sauvegardés: {json_path}")
    
    # 3. Sauvegarder le meilleur config en pickle
    best_path = output_dir / f'best_config_{name_result}.pkl'
    with open(best_path, 'wb') as f:
        pickle.dump(best_config, f)
    
    print(f"✅ Meilleure configuration sauvegardée: {best_path}")
    
    # 4. Sauvegarder un rapport texte
    report_path = output_dir / f'cv_report_{name_result}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RAPPORT DE VALIDATION CROISÉE\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre de configurations testées: {len(all_results)}\n\n")
        
        f.write("MEILLEURE CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"  alpha: {best_config.alpha}\n")
        f.write(f"  beta: {best_config.beta}\n")
        f.write(f"  orphan_fraction: {best_config.orphan_fraction}\n")
        f.write(f"  tau: {best_config.tau}\n")
        f.write(f"  gnn_layers: {best_config.gnn_layers}\n")
        f.write(f"  lr: {best_config.lr}\n")
        f.write(f"  dropout: {best_config.dropout}\n")
        f.write(f"  weight_decay: {best_config.weight_decay}\n\n")
        
        # Top 10 des configurations
        f.write("TOP 10 CONFIGURATIONS (par Alignment):\n")
        f.write("-"*40 + "\n")
        
        sorted_configs = sorted(
            all_results.items(),
            key=lambda x: x[1]['metrics']['avg_alignment']
        )[:10]
        
        for i, (config_id, results) in enumerate(sorted_configs, 1):
            params = results['params']
            metrics = results['metrics']
            f.write(f"\n{i}. Config {config_id}:\n")
            f.write(f"   alpha={params['alpha']}, beta={params['beta']}, orphan={params['orphan_fraction']}\n")
            f.write(f"   tau={params.get('tau', 0.1)}, layers={params['gnn_layers']}\n")
            f.write(f"   Alignment: {metrics['avg_alignment']:.4f}, Uniformity: {metrics['avg_uniformity']:.4f}\n")
            f.write(f"   Best loss: {metrics['best_loss']:.4f}\n")
    
    print(f"✅ Rapport sauvegardé: {report_path}")
    
    # 5. Statistiques globales
    stats_path = output_dir / f'cv_statistics_{name_result}.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STATISTIQUES DE LA VALIDATION CROISÉE\n")
        f.write("="*60 + "\n\n")
        
        # Extraire toutes les métriques
        alignments = [r['metrics']['avg_alignment'] for r in all_results.values() 
                      if r['metrics']['avg_alignment'] != float('inf')]
        uniformities = [r['metrics']['avg_uniformity'] for r in all_results.values()
                        if r['metrics']['avg_uniformity'] != float('inf')]
        
        if alignments:
            f.write(f"Nombre de configurations valides: {len(alignments)}\n")
            f.write(f"Alignment - min: {min(alignments):.4f}\n")
            f.write(f"Alignment - max: {max(alignments):.4f}\n")
            f.write(f"Alignment - mean: {sum(alignments)/len(alignments):.4f}\n")
            f.write(f"Alignment - std: {(sum((a - sum(alignments)/len(alignments))**2 for a in alignments)/len(alignments))**0.5:.4f}\n\n")
        
        if uniformities:
            f.write(f"Uniformity - min: {min(uniformities):.4f}\n")
            f.write(f"Uniformity - max: {max(uniformities):.4f}\n")
            f.write(f"Uniformity - mean: {sum(uniformities)/len(uniformities):.4f}\n")
            f.write(f"Uniformity - std: {(sum((u - sum(uniformities)/len(uniformities))**2 for u in uniformities)/len(uniformities))**0.5:.4f}\n")
    
    print(f"✅ Statistiques sauvegardées: {stats_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_cv_main():
    """Script principal de validation croisée."""
    parser = argparse.ArgumentParser(description="Validation croisée")
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Chemin vers la configuration de base'
    )
    parser.add_argument(
        '--cv_config',
        type=str,
        default='config/cv.yaml',
        help='Chemin vers la configuration de la CV'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='./data/graphs/zulu',
        help='Dossier des données pré-traitées'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/zulu/cv',
        help='Dossier de sortie'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Mode rapide'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=None,
        help='Nombre de folds'
    )
    args = parser.parse_args()
    
    # Charger les configurations
    print("📂 Chargement des configurations...")
    
    with open(args.cv_config, 'r') as f:
        cv_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Configurer le mode rapide
    epochs = int(cv_config.get('epochs', 20))
    if args.quick:
        epochs = 5
        cv_config['alpha_grid'] = [0.001]
        cv_config['beta_grid'] = [0.001]
        cv_config['orphan_fraction_grid'] = [0.1]
        cv_config['gnn_layers_grid'] = [3, 4, 5]
        cv_config['lr_grid'] = [0.001]
        cv_config['dropout_grid'] = [0.1]
        cv_config['weight_decay_grid'] = [1e-5]
        cv_config['tau_grid'] = [0.1, 0.3, 0.7]
        print("⚡ Mode rapide activé")
    
    n_folds = args.folds or cv_config.get('n_folds', 2)
    
    print("="*60)
    print("🔬 VALIDATION CROISÉE")
    print("="*60)
    print(f"📁 Processed dir: {args.processed_dir}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"📊 Folds: {n_folds}")
    print(f"📊 Époques par fold: {epochs}")
    
    # Vérifier que les données existent
    model = base_config['encoder']['backbone'].split('/')[1]
    representation = base_config['linguistic']['representation']
    features_path = Path(args.processed_dir) / 'features' / f'frame_features_{model}.pt'
    if not features_path.exists():
        print(f"❌ Fichier non trouvé: {features_path}")
        print("   Veuillez d'abord exécuter build_save_graph.py")
        return
    
    # Charger les données pour les splits
    frame_features = torch.load(features_path, map_location='cpu')
    n_examples = frame_features.shape[0]
    print(f"📊 {n_examples} exemples chargés")
    
    # Créer les folds
    folds = create_cv_folds(
        n_examples=n_examples,
        n_folds=n_folds,
        by_speaker=False,
        seed=cv_config.get('seed', 42)
    )
    print(f"📊 {len(folds)} folds créés")
    
    # 🔑 Dictionnaire pour stocker tous les résultats
    all_results = {}
    config_counter = 0
    
    # Fonction pour la CV (1 argument)
    def train_and_eval(config: HyperparamConfig):
        """Wrapper pour trainer une configuration."""
        nonlocal config_counter
        config_counter += 1
        config_id = f"config_{config_counter:04d}"
        
        print(f"\n🔍 Test {config_counter}:")
        print(f"   alpha={config.alpha}, beta={config.beta}, orphan={config.orphan_fraction}")
        print(f"   tau={config.tau}, layers={config.gnn_layers}")
        
        results = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"\n📁 Fold {fold+1}/{len(folds)}")
            print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")
            
            result = train_and_evaluate_config(
                hyperparams=config,
                base_config=base_config,
                processed_dir=args.processed_dir,
                output_dir=args.output_dir,
                fold=fold,
                train_indices=train_idx,
                val_indices=val_idx,
                epochs=epochs,
                default_config_path=args.config
            )
            results.append(result)
            fold_results.append({
                'fold': fold,
                'alignment': result['alignment'],
                'uniformity': result['uniformity'],
                'best_loss': result['best_loss']
            })
        
        # Moyenne sur les folds
        valid_results = [r for r in results if r['alignment'] != float('inf')]
        
        if not valid_results:
            print(f"   ❌ Tous les folds ont échoué pour cette configuration")
            all_results[config_id] = {
                'params': config.to_dict(),
                'metrics': {
                    'avg_alignment': float('inf'),
                    'avg_uniformity': float('inf'),
                    'best_loss': float('inf'),
                    'best_epoch': -1,
                    'num_folds_success': 0
                },
                'fold_results': fold_results,
                'is_best': False
            }
            return {
                'alignment': float('inf'),
                'uniformity': float('inf'),
                'link_pred_outputs': (torch.tensor([]), torch.tensor([]))
            }
        
        avg_alignment = sum(r['alignment'] for r in valid_results) / len(valid_results)
        avg_uniformity = sum(r['uniformity'] for r in valid_results) / len(valid_results)
        avg_best_loss = sum(r['best_loss'] for r in valid_results) / len(valid_results)
        
        all_p_pred = torch.cat([r['link_pred_outputs'][0] for r in valid_results if r['link_pred_outputs'][0].numel() > 0], dim=0)
        all_y_true = torch.cat([r['link_pred_outputs'][1] for r in valid_results if r['link_pred_outputs'][1].numel() > 0], dim=0)
        
        print(f"\n📊 Résultats moyens:")
        print(f"   Alignment: {avg_alignment:.4f}")
        print(f"   Uniformity: {avg_uniformity:.4f}")
        print(f"   Best Loss: {avg_best_loss:.4f}")
        
        # Stocker les résultats
        all_results[config_id] = {
            'params': config.to_dict(),
            'metrics': {
                'avg_alignment': avg_alignment,
                'avg_uniformity': avg_uniformity,
                'best_loss': avg_best_loss,
                'best_epoch': -1,
                'num_folds_success': len(valid_results)
            },
            'fold_results': fold_results,
            'is_best': False
        }
        
        return {
            'alignment': avg_alignment,
            'uniformity': avg_uniformity,
            'link_pred_outputs': (all_p_pred, all_y_true)
        }
    
    # Lancer la CV
    try:
        best_config = run_cross_validation(
            cv_config=cv_config,
            train_and_eval_fn=train_and_eval
        )
        
        if best_config is None:
            print("❌ Aucune configuration valide trouvée")
            return
        
        # Marquer la meilleure configuration
        for config_id, results in all_results.items():
            params = results['params']
            if (params['alpha'] == best_config.alpha and
                params['beta'] == best_config.beta and
                params['orphan_fraction'] == best_config.orphan_fraction and
                params['gnn_layers'] == best_config.gnn_layers and
                params.get('tau') == best_config.tau):
                results['is_best'] = True
                break
        
        # Sauvegarder tous les résultats
        save_all_results(
            all_results=all_results,
            output_dir=Path(args.output_dir),
            best_config=best_config,
            name_result=f'{model}_{representation}'
        )
        
        print("\n" + "="*60)
        print("📝 CONFIGURATION POUR L'ENTRAÎNEMENT FINAL")
        print("="*60)
        print(f"""
Mettez à jour config/default.yaml avec ces valeurs:
  loss_weights:
    alpha: {best_config.alpha}
    beta: {best_config.beta}
  dropout:
    orphan_fraction: {best_config.orphan_fraction}
  graph:
    gnn_layers: {best_config.gnn_layers}
  training:
    lr_gnn: {best_config.lr}
    dropout: {best_config.dropout}
    weight_decay: {best_config.weight_decay}
    tau: {best_config.tau}
        """)
        
    except Exception as e:
        print(f"❌ Erreur lors de la CV: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_cv_main()
