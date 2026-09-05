"""
analyze_gnn.py
Analyse la taille et les paramètres du GNN.
À placer dans le dossier racine du projet.
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier racine du projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"📁 Project root: {project_root}")

import torch
from models.gnn import HeterogeneousGNN


def count_parameters(model):
    """Compte le nombre total de paramètres et de paramètres entraînables."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def print_model_structure(model, indent=0):
    """Affiche la structure du modèle de manière récursive."""
    prefix = "  " * indent
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{prefix}├── {name}: {module.__class__.__name__} ({num_params:,} paramètres)")
        if len(list(module.children())) > 0:
            print_model_structure(module, indent + 1)


def create_edge_index(num_nodes_a: int, num_nodes_w: int, num_edges: int = 20):
    """
    Crée des edge indices valides pour l'analyse.
    
    Args:
        num_nodes_a: Nombre de nœuds audio
        num_nodes_w: Nombre de nœuds word
        num_edges: Nombre d'arêtes à créer
    """
    # Arêtes audio-audio
    audio_audio = torch.randint(0, num_nodes_a, (2, num_edges))
    
    # Arêtes word-word
    word_word = torch.randint(0, num_nodes_w, (2, num_edges))
    
    # Arêtes audio-word (cross)
    audio_word = torch.randint(0, num_nodes_a, (1, num_edges))
    audio_word = torch.cat([audio_word, torch.randint(0, num_nodes_w, (1, num_edges))], dim=0)
    
    # Arêtes word-audio (cross inverse)
    word_audio = torch.randint(0, num_nodes_w, (1, num_edges))
    word_audio = torch.cat([word_audio, torch.randint(0, num_nodes_a, (1, num_edges))], dim=0)
    
    return {
        ('audio', 'similar_to', 'audio'): audio_audio,
        ('word', 'similar_to', 'word'): word_word,
        ('audio', 'transcribed_as', 'word'): audio_word,
        ('word', 'rev_transcribed_as', 'audio'): word_audio,
    }


def analyze_gnn(hidden_dim: int = 768, n_layers: int = 2, dropout: float = 0.1, 
                audio_dim: int = 1024, linguistic_dim: int = 768,
                num_audio: int = 10, num_words: int = 5):
    """
    Analyse le GNN avec les dimensions spécifiées.
    
    Args:
        hidden_dim: Dimension cachée (ex: 768)
        n_layers: Nombre de couches GNN
        dropout: Taux de dropout
        audio_dim: Dimension des features audio (ex: 1024 pour wav2vec2)
        linguistic_dim: Dimension des features linguistiques (ex: 768 pour LaBSE)
        num_audio: Nombre de nœuds audio factices
        num_words: Nombre de nœuds word factices
    """
    print("="*60)
    print("ANALYSE DU GNN")
    print("="*60)
    print(f"🔧 Configuration:")
    print(f"   - Hidden dimension: {hidden_dim}")
    print(f"   - Nombre de couches: {n_layers}")
    print(f"   - Dropout: {dropout}")
    print(f"   - Audio dimension: {audio_dim}")
    print(f"   - Linguistic dimension: {linguistic_dim}")
    print(f"   - Nœuds audio factices: {num_audio}")
    print(f"   - Nœuds word factices: {num_words}")
    
    # Créer le modèle
    model = HeterogeneousGNN(
        n_layers=n_layers,
        dropout=dropout
    )
    
    # 🔑 Créer des features factices avec les bonnes dimensions
    seq_len = 195
    
    features_dict = {
        'audio': torch.randn(num_audio, seq_len, audio_dim),
        'word': torch.randn(num_words, linguistic_dim)
    }
    attention_mask_dict = {
        'audio': None,
        'word': None
    }
    
    # 🔑 Créer des edge indices valides
    edge_index_dict = create_edge_index(num_audio, num_words)
    
    # Vérifier les edge indices
    print(f"\n🔍 Vérification des edge indices:")
    for key, edge_index in edge_index_dict.items():
        max_idx = edge_index.max().item()
        print(f"   - {key}: max index = {max_idx}")
    
    # Forward pour initialiser
    print(f"\n🔄 Initialisation des couches dynamiques...")
    with torch.no_grad():
        try:
            model(features_dict, attention_mask_dict, edge_index_dict)
            print("   ✅ Initialisation réussie")
        except Exception as e:
            print(f"   ⚠️ Erreur lors de l'initialisation: {e}")
            # Fallback: initialiser manuellement avec des dimensions par défaut
            model._initialize_dynamic_layers(hidden_dim, audio_dim, linguistic_dim)
            print(f"   ✅ Initialisation forcée avec hidden_dim={hidden_dim}")
    
    print(f"\n📊 Statistiques du modèle:")
    print(f"   - Hidden dimension: {model.hidden_dim}")
    
    # Compter les paramètres
    params = count_parameters(model)
    print(f"\n📈 Paramètres:")
    print(f"   - Total: {params['total']:,}")
    print(f"   - Entraînables: {params['trainable']:,}")
    print(f"   - Non entraînables: {params['non_trainable']:,}")
    
    # Taille approximative en mémoire
    param_size = params['trainable'] * 4 / (1024 * 1024)  # 4 octets par paramètre (float32)
    print(f"\n💾 Taille approximative: {param_size:.2f} MB (en float32)")
    
    # Structure du modèle
    print(f"\n📐 Structure du modèle:")
    print_model_structure(model)
    
    # Détail par couche avec paramètres
    print(f"\n🔍 Détail des couches entraînables:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            shape_str = ' × '.join([str(s) for s in param.shape])
            print(f"   - {name}: {shape_str} → {num_params:,} paramètres")
    
    print(f"\n✅ Total paramètres entraînables: {total_params:,}")
    
    return model, params


def analyze_with_different_dimensions():
    """Analyse le GNN avec différentes dimensions."""
    print("\n" + "="*60)
    print("COMPARAISON DES DIMENSIONS")
    print("="*60)
    
    configs = [
        {'hidden_dim': 256, 'n_layers': 2},
        {'hidden_dim': 384, 'n_layers': 2},
        {'hidden_dim': 512, 'n_layers': 2},
        {'hidden_dim': 768, 'n_layers': 2},
        {'hidden_dim': 1024, 'n_layers': 2},
        {'hidden_dim': 768, 'n_layers': 3},
        {'hidden_dim': 768, 'n_layers': 4},
    ]
    
    print(f"\n{'Hidden Dim':<12} {'Layers':<8} {'Paramètres':<15} {'Taille (MB)':<12}")
    print("-"*50)
    
    for config in configs:
        hidden_dim = config['hidden_dim']
        n_layers = config['n_layers']
        
        # Créer le modèle
        model = HeterogeneousGNN(n_layers=n_layers, dropout=0.1)
        
        # Forward pour initialiser
        audio_dim = hidden_dim
        linguistic_dim = hidden_dim
        num_audio = 10
        num_words = 5
        
        features_dict = {
            'audio': torch.randn(num_audio, 195, audio_dim),
            'word': torch.randn(num_words, linguistic_dim)
        }
        edge_index_dict = create_edge_index(num_audio, num_words)
        
        with torch.no_grad():
            try:
                model(features_dict, None, edge_index_dict)
            except Exception as e:
                print(f"   ⚠️ {e}")
                model._initialize_dynamic_layers(hidden_dim, audio_dim, linguistic_dim)
        
        params = count_parameters(model)
        size_mb = params['trainable'] * 4 / (1024 * 1024)
        
        print(f"{hidden_dim:<12} {n_layers:<8} {params['trainable']:>12,}   {size_mb:>8.2f} MB")


def main():
    """Script principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse du GNN")
    parser.add_argument('--hidden_dim', type=int, default=768, help='Dimension cachée')
    parser.add_argument('--n_layers', type=int, default=2, help='Nombre de couches')
    parser.add_argument('--dropout', type=float, default=0.1, help='Taux de dropout')
    parser.add_argument('--audio_dim', type=int, default=1024, help='Dimension audio (wav2vec2)')
    parser.add_argument('--linguistic_dim', type=int, default=768, help='Dimension linguistique (LaBSE)')
    parser.add_argument('--num_audio', type=int, default=10, help='Nombre de nœuds audio factices')
    parser.add_argument('--num_words', type=int, default=5, help='Nombre de nœuds word factices')
    parser.add_argument('--compare', action='store_true', help='Comparer les dimensions')
    
    args = parser.parse_args()
    
    if args.compare:
        analyze_with_different_dimensions()
    else:
        analyze_gnn(
            args.hidden_dim, 
            args.n_layers, 
            args.dropout,
            args.audio_dim, 
            args.linguistic_dim,
            args.num_audio,
            args.num_words
        )


if __name__ == "__main__":
    main()
