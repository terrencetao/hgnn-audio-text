"""
get_best_config.py
Récupère la meilleure configuration de la validation croisée.
Peut créer un nouveau fichier config si nécessaire.
"""

import pickle
import yaml
import shutil
import argparse
from pathlib import Path
from dataclasses import asdict


def get_best_config(output_dir="./outputs/cv"):
    """Récupère la meilleure configuration depuis les résultats de la CV."""
    output_dir = Path(output_dir)
    
    # 1. Depuis le fichier pickle
    best_config_path = output_dir / 'best_config.pkl'
    if best_config_path.exists():
        with open(best_config_path, 'rb') as f:
            best_config = pickle.load(f)
        
        print("="*60)
        print("MEILLEURE CONFIGURATION (depuis pickle)")
        print("="*60)
        print(f"  alpha: {best_config.alpha}")
        print(f"  beta: {best_config.beta}")
        print(f"  orphan_fraction: {best_config.orphan_fraction}")
        print(f"  tau: {best_config.tau}")
        print(f"  gnn_layers: {best_config.gnn_layers}")
        print(f"  lr: {best_config.lr}")
        print(f"  dropout: {best_config.dropout}")
        print(f"  weight_decay: {best_config.weight_decay}")
        print()
        
        return best_config
    
    # 2. Depuis le CSV
    csv_path = output_dir / 'all_results.csv'
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        best_row = df[df['is_best'] == 'YES']
        if not best_row.empty:
            print("="*60)
            print("MEILLEURE CONFIGURATION (depuis CSV)")
            print("="*60)
            row = best_row.iloc[0]
            print(f"  alpha: {row['alpha']}")
            print(f"  beta: {row['beta']}")
            print(f"  orphan_fraction: {row['orphan_fraction']}")
            print(f"  tau: {row['tau']}")
            print(f"  gnn_layers: {row['gnn_layers']}")
            print(f"  lr: {row['lr']}")
            print(f"  dropout: {row['dropout']}")
            print(f"  weight_decay: {row['weight_decay']}")
            print(f"  avg_alignment: {row['avg_alignment']:.4f}")
            print(f"  avg_uniformity: {row['avg_uniformity']:.4f}")
            print()
            
            from cross_validation import HyperparamConfig
            return HyperparamConfig(
                alpha=float(row['alpha']),
                beta=float(row['beta']),
                orphan_fraction=float(row['orphan_fraction']),
                tau=float(row['tau']) if pd.notna(row['tau']) else 0.1,
                gnn_layers=int(row['gnn_layers']),
                lr=float(row['lr']),
                dropout=float(row['dropout']),
                weight_decay=float(row['weight_decay'])
            )
    
    print("❌ Aucun résultat trouvé")
    return None


def create_default_config(config_path="config/default.yaml"):
    """
    Crée un fichier de configuration par défaut s'il n'existe pas.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        'data': {
            'raw_dir': './data/raw/ewondo',
            'metadata_csv': './data/processed/ewondo/metadata.csv',
            'processed_dir': './data/graphs/ewondo',
            'sample_rate': 16000,
            'max_length': 160000,
            'normalize': True
        },
        'encoder': {
            'backbone': 'facebook/wav2vec2-xls-r-300m',
            'hidden_dim': 768,
            'freeze_backbone': True
        },
        'linguistic': {
            'representation': 'labse'
        },
        'graph': {
            'build': {
                'similarity_threshold': 0.6,
                'use_mst': True,
                'backbone_method': 'mst',
                'ensure_connectivity': True,
                'pooling_for_graph': 'mean'
            },
            'gnn_layers': 2
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'lr_gnn': 0.001,
            'weight_decay': 1e-5,
            'dropout': 0.1,
            'grad_clip': 1.0,
            'device': 'cpu',
            'seed': 42,
            'output_dir': './outputs/ewondo',
            'test_edge_fraction': 0.1
        },
        'loss_weights': {
            'alpha': 0.5,
            'beta': 0.5
        },
        'dropout': {
            'orphan_fraction': 0.3
        },
        'system': {
            'num_workers': 4,
            'pin_memory': True,
            'verbose': True
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Fichier config créé: {config_path}")
    return default_config


def update_config_with_best(
    config_path="config/default.yaml", 
    output_dir="./outputs/cv",
    create_if_missing=True
):
    """
    Met à jour le fichier de config avec la meilleure configuration.
    Crée le fichier s'il n'existe pas.
    """
    config_path = Path(config_path)
    
    # Vérifier si le fichier config existe
    if not config_path.exists():
        if create_if_missing:
            print(f"⚠️ Fichier config non trouvé: {config_path}")
            print("   Création d'un fichier par défaut...")
            default_config = create_default_config(str(config_path))
            config = default_config
        else:
            raise FileNotFoundError(f"Fichier config non trouvé: {config_path}")
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # Récupérer la meilleure configuration
    best_config = get_best_config(output_dir)
    if best_config is None:
        print("❌ Aucune meilleure configuration trouvée")
        return None
    
    # Mettre à jour la config
    # Loss weights
    if 'loss_weights' not in config:
        config['loss_weights'] = {}
    config['loss_weights']['alpha'] = best_config.alpha
    config['loss_weights']['beta'] = best_config.beta
    
    # Dropout
    if 'dropout' not in config:
        config['dropout'] = {}
    config['dropout']['orphan_fraction'] = best_config.orphan_fraction
    
    # Graph
    if 'graph' not in config:
        config['graph'] = {}
    config['graph']['gnn_layers'] = best_config.gnn_layers
    
    # Training
    if 'training' not in config:
        config['training'] = {}
    config['training']['lr_gnn'] = best_config.lr
    config['training']['dropout'] = best_config.dropout
    config['training']['weight_decay'] = best_config.weight_decay
    
    # 🔑 Ajouter tau si présent
    if hasattr(best_config, 'tau') and best_config.tau is not None:
        if 'loss_weights' not in config:
            config['loss_weights'] = {}
        config['loss_weights']['tau'] = best_config.tau
    
    # Sauvegarder le backup
    backup_path = config_path.with_suffix('.yaml.backup')
    shutil.copy2(config_path, backup_path)
    print(f"✅ Backup sauvegardé: {backup_path}")
    
    # Sauvegarder la config mise à jour
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ Fichier config mis à jour: {config_path}")
    print(f"\n📝 Nouvelles valeurs:")
    print(f"   loss_weights:")
    print(f"     alpha: {best_config.alpha}")
    print(f"     beta: {best_config.beta}")
    if hasattr(best_config, 'tau') and best_config.tau is not None:
        print(f"     tau: {best_config.tau}")
    print(f"   dropout:")
    print(f"     orphan_fraction: {best_config.orphan_fraction}")
    print(f"   graph:")
    print(f"     gnn_layers: {best_config.gnn_layers}")
    print(f"   training:")
    print(f"     lr_gnn: {best_config.lr}")
    print(f"     dropout: {best_config.dropout}")
    print(f"     weight_decay: {best_config.weight_decay}")
    
    return config


def main():
    """Script principal."""
    parser = argparse.ArgumentParser(
        description="Récupère la meilleure configuration de la CV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/cv",
        help="Dossier des résultats de la CV (défaut: ./outputs/ewondo/cv)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Chemin vers le fichier de configuration (défaut: config/default.yaml)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Mettre à jour le fichier de configuration"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Créer le fichier de configuration s'il n'existe pas"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Afficher la meilleure configuration"
    )
    
    args = parser.parse_args()
    
    # Mode update
    if args.update:
        update_config_with_best(
            config_path=args.config,
            output_dir=args.output_dir,
            create_if_missing=args.create
        )
    else:
        # Mode show (par défaut)
        best_config = get_best_config(args.output_dir)
        if best_config:
            print("\n💡 Pour mettre à jour votre config:")
            print(f"   python get_best_config.py --update --config {args.config}")
            if args.create:
                print("   (le fichier sera créé automatiquement s'il n'existe pas)")


if __name__ == "__main__":
    main()
