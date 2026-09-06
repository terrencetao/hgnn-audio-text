"""
analyze_graph.py

Script complet d'analyse du graphe hétérogène.
Fournit des statistiques détaillées sur :
- Structure du graphe (nœuds, arêtes, degrés)
- Connectivité et composantes
- Distribution des similarités
- Qualité des arêtes cross
- Statistiques par nœud audio et word
- Visualisation des distributions
"""
import sys
from pathlib import Path
# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Any
import json
import warnings
warnings.filterwarnings("ignore")

from graph.build_graph import GraphBuildConfig
from graph.similarity import (
    LinguisticRepresentation,
    compute_labse_embeddings,
    compute_bag_of_phonemes,
    compute_articulatory_features
)


class GraphAnalyzer:
    """
    Analyseur complet de graphe hétérogène.
    """
    
    def __init__(
        self,
        graph_path: str,
        config_path: str = None,
        output_dir: str = "./graph_analysis"
    ):
        self.graph_path = Path(graph_path)
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger le graphe
        print(f"📂 Chargement du graphe: {graph_path}")
        self.graph = torch.load(graph_path, map_location='cpu')
        
        # Charger la config si fournie
        self.config = None
        if config_path:
            self.config = OmegaConf.load(config_path)
        
        # Extraire les informations
        self._extract_graph_info()
        
        print(f"\n✅ Graphe chargé avec succès!")
        print(f"   - Nœuds audio: {self.n_audio}")
        print(f"   - Nœuds word: {self.n_word}")
        print(f"   - Arêtes audio-audio: {self.n_audio_audio_edges}")
        print(f"   - Arêtes word-word: {self.n_word_word_edges}")
        print(f"   - Arêtes cross: {self.n_cross_edges}")
    
    def _extract_graph_info(self):
        """Extrait les informations de base du graphe."""
        # Nombre de nœuds
        self.n_audio = self.graph["audio"].x.shape[0]
        self.n_word = self.graph["word"].x.shape[0]
        
        # Features
        self.audio_features = self.graph["audio"].x
        self.word_features = self.graph["word"].x
        self.audio_dim = self.audio_features.shape[1]
        self.word_dim = self.word_features.shape[1]
        
        # Arêtes audio-audio
        if ("audio", "similar_to", "audio") in self.graph.edge_index_dict:
            self.audio_audio_edges = self.graph["audio", "similar_to", "audio"].edge_index
            self.n_audio_audio_edges = self.audio_audio_edges.shape[1]
            self.audio_audio_weights = self.graph["audio", "similar_to", "audio"].edge_weight if hasattr(self.graph["audio", "similar_to", "audio"], 'edge_weight') else None
        else:
            self.audio_audio_edges = None
            self.n_audio_audio_edges = 0
            self.audio_audio_weights = None
        
        # Arêtes word-word
        if ("word", "similar_to", "word") in self.graph.edge_index_dict:
            self.word_word_edges = self.graph["word", "similar_to", "word"].edge_index
            self.n_word_word_edges = self.word_word_edges.shape[1]
        else:
            self.word_word_edges = None
            self.n_word_word_edges = 0
        
        # Arêtes cross
        if ("audio", "transcribed_as", "word") in self.graph.edge_index_dict:
            self.cross_edges = self.graph["audio", "transcribed_as", "word"].edge_index
            self.n_cross_edges = self.cross_edges.shape[1]
            self.cross_weights = self.graph["audio", "transcribed_as", "word"].edge_weight if hasattr(self.graph["audio", "transcribed_as", "word"], 'edge_weight') else None
        else:
            self.cross_edges = None
            self.n_cross_edges = 0
            self.cross_weights = None
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        Analyse la structure du graphe.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DE LA STRUCTURE DU GRAPHE")
        print("="*60)
        
        results = {
            "nodes": {
                "audio": self.n_audio,
                "word": self.n_word,
                "total": self.n_audio + self.n_word
            },
            "edges": {
                "audio_audio": self.n_audio_audio_edges,
                "word_word": self.n_word_word_edges,
                "cross": self.n_cross_edges,
                "total": self.n_audio_audio_edges + self.n_word_word_edges + self.n_cross_edges
            },
            "features": {
                "audio_dim": self.audio_dim,
                "word_dim": self.word_dim
            }
        }
        
        # Densité du graphe
        max_audio_edges = self.n_audio * (self.n_audio - 1) / 2
        max_word_edges = self.n_word * (self.n_word - 1) / 2
        max_cross_edges = self.n_audio * self.n_word
        
        results["density"] = {
            "audio_audio": self.n_audio_audio_edges / max_audio_edges if max_audio_edges > 0 else 0,
            "word_word": self.n_word_word_edges / max_word_edges if max_word_edges > 0 else 0,
            "cross": self.n_cross_edges / max_cross_edges if max_cross_edges > 0 else 0
        }
        
        print(f"\n📈 Nœuds:")
        print(f"   - Audio: {self.n_audio}")
        print(f"   - Word: {self.n_word}")
        print(f"   - Total: {self.n_audio + self.n_word}")
        
        print(f"\n🔗 Arêtes:")
        print(f"   - Audio-Audio: {self.n_audio_audio_edges}")
        print(f"   - Word-Word: {self.n_word_word_edges}")
        print(f"   - Cross: {self.n_cross_edges}")
        print(f"   - Total: {self.n_audio_audio_edges + self.n_word_word_edges + self.n_cross_edges}")
        
        print(f"\n📊 Densité:")
        print(f"   - Audio-Audio: {results['density']['audio_audio']:.4f}")
        print(f"   - Word-Word: {results['density']['word_word']:.4f}")
        print(f"   - Cross: {results['density']['cross']:.4f}")
        
        return results
    
    def analyze_degrees(self) -> Dict[str, Any]:
        """
        Analyse les degrés des nœuds.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DES DEGRÉS")
        print("="*60)
        
        results = {}
        
        # Degrés audio-audio
        if self.audio_audio_edges is not None and self.n_audio_audio_edges > 0:
            audio_degrees = torch.zeros(self.n_audio, dtype=torch.long)
            for i in range(self.n_audio_audio_edges):
                audio_degrees[self.audio_audio_edges[0, i]] += 1
                audio_degrees[self.audio_audio_edges[1, i]] += 1
            
            results["audio_audio_degrees"] = {
                "mean": audio_degrees.float().mean().item(),
                "std": audio_degrees.float().std().item(),
                "min": audio_degrees.min().item(),
                "max": audio_degrees.max().item(),
                "median": audio_degrees.median().item(),
                "distribution": Counter(audio_degrees.tolist())
            }
            
            print(f"\n🎵 Degrés Audio-Audio:")
            print(f"   - Moyenne: {results['audio_audio_degrees']['mean']:.2f}")
            print(f"   - Écart-type: {results['audio_audio_degrees']['std']:.2f}")
            print(f"   - Min/Max: {results['audio_audio_degrees']['min']} / {results['audio_audio_degrees']['max']}")
            print(f"   - Médiane: {results['audio_audio_degrees']['median']}")
        
        # Degrés cross (audio vers word)
        if self.cross_edges is not None and self.n_cross_edges > 0:
            audio_cross_degrees = torch.zeros(self.n_audio, dtype=torch.long)
            word_cross_degrees = torch.zeros(self.n_word, dtype=torch.long)
            
            for i in range(self.n_cross_edges):
                audio_idx = self.cross_edges[0, i].item()
                word_idx = self.cross_edges[1, i].item()
                audio_cross_degrees[audio_idx] += 1
                word_cross_degrees[word_idx] += 1
            
            results["cross_degrees_audio"] = {
                "mean": audio_cross_degrees.float().mean().item(),
                "std": audio_cross_degrees.float().std().item(),
                "min": audio_cross_degrees.min().item(),
                "max": audio_cross_degrees.max().item(),
                "median": audio_cross_degrees.median().item(),
                "distribution": Counter(audio_cross_degrees.tolist())
            }
            
            results["cross_degrees_word"] = {
                "mean": word_cross_degrees.float().mean().item(),
                "std": word_cross_degrees.float().std().item(),
                "min": word_cross_degrees.min().item(),
                "max": word_cross_degrees.max().item(),
                "median": word_cross_degrees.median().item(),
                "distribution": Counter(word_cross_degrees.tolist())
            }
            
            print(f"\n🔗 Degrés Cross (Audio vers Word):")
            print(f"   - Audio - Moyenne: {results['cross_degrees_audio']['mean']:.2f}")
            print(f"   - Audio - Min/Max: {results['cross_degrees_audio']['min']} / {results['cross_degrees_audio']['max']}")
            print(f"   - Word - Moyenne: {results['cross_degrees_word']['mean']:.2f}")
            print(f"   - Word - Min/Max: {results['cross_degrees_word']['min']} / {results['cross_degrees_word']['max']}")
        
        return results
    
    def analyze_weights(self) -> Dict[str, Any]:
        """
        Analyse les poids des arêtes.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DES POIDS")
        print("="*60)
        
        results = {}
        
        # Poids audio-audio
        if self.audio_audio_weights is not None and len(self.audio_audio_weights) > 0:
            weights = self.audio_audio_weights.numpy()
            results["audio_audio_weights"] = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "median": float(np.median(weights)),
                "percentiles": {
                    "25": float(np.percentile(weights, 25)),
                    "50": float(np.percentile(weights, 50)),
                    "75": float(np.percentile(weights, 75)),
                    "90": float(np.percentile(weights, 90)),
                    "95": float(np.percentile(weights, 95))
                }
            }
            
            print(f"\n🎵 Poids Audio-Audio:")
            print(f"   - Moyenne: {results['audio_audio_weights']['mean']:.4f}")
            print(f"   - Écart-type: {results['audio_audio_weights']['std']:.4f}")
            print(f"   - Min/Max: {results['audio_audio_weights']['min']:.4f} / {results['audio_audio_weights']['max']:.4f}")
            print(f"   - Médiane: {results['audio_audio_weights']['median']:.4f}")
            print(f"   - Percentiles: 25%={results['audio_audio_weights']['percentiles']['25']:.4f}, "
                  f"95%={results['audio_audio_weights']['percentiles']['95']:.4f}")
        
        # Poids cross
        if self.cross_weights is not None and len(self.cross_weights) > 0:
            weights = self.cross_weights.numpy()
            results["cross_weights"] = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "median": float(np.median(weights)),
                "percentiles": {
                    "25": float(np.percentile(weights, 25)),
                    "50": float(np.percentile(weights, 50)),
                    "75": float(np.percentile(weights, 75)),
                    "90": float(np.percentile(weights, 90)),
                    "95": float(np.percentile(weights, 95))
                },
                "exact_match_count": int((weights == 1.0).sum()),
                "exact_match_fraction": float((weights == 1.0).sum() / len(weights))
            }
            
            print(f"\n🔗 Poids Cross:")
            print(f"   - Moyenne: {results['cross_weights']['mean']:.4f}")
            print(f"   - Écart-type: {results['cross_weights']['std']:.4f}")
            print(f"   - Min/Max: {results['cross_weights']['min']:.4f} / {results['cross_weights']['max']:.4f}")
            print(f"   - Médiane: {results['cross_weights']['median']:.4f}")
            print(f"   - Correspondances exactes (poids=1.0): {results['cross_weights']['exact_match_count']} ({results['cross_weights']['exact_match_fraction']*100:.2f}%)")
        
        return results
    
    def analyze_connectivity(self) -> Dict[str, Any]:
        """
        Analyse la connectivité du graphe.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DE LA CONNECTIVITÉ")
        print("="*60)
        
        results = {}
        
        # Construire le graphe NetworkX pour l'analyse
        G = nx.Graph()
        
        # Ajouter les nœuds audio
        G.add_nodes_from([(f"a{i}", {"type": "audio"}) for i in range(self.n_audio)])
        G.add_nodes_from([(f"w{i}", {"type": "word"}) for i in range(self.n_word)])
        
        # Ajouter les arêtes audio-audio
        if self.audio_audio_edges is not None:
            for i in range(self.n_audio_audio_edges):
                u = f"a{self.audio_audio_edges[0, i].item()}"
                v = f"a{self.audio_audio_edges[1, i].item()}"
                weight = self.audio_audio_weights[i].item() if self.audio_audio_weights is not None else 1.0
                G.add_edge(u, v, weight=weight, type="audio_audio")
        
        # Ajouter les arêtes word-word
        if self.word_word_edges is not None:
            for i in range(self.n_word_word_edges):
                u = f"w{self.word_word_edges[0, i].item()}"
                v = f"w{self.word_word_edges[1, i].item()}"
                G.add_edge(u, v, weight=1.0, type="word_word")
        
        # Ajouter les arêtes cross
        if self.cross_edges is not None:
            for i in range(self.n_cross_edges):
                u = f"a{self.cross_edges[0, i].item()}"
                v = f"w{self.cross_edges[1, i].item()}"
                weight = self.cross_weights[i].item() if self.cross_weights is not None else 1.0
                G.add_edge(u, v, weight=weight, type="cross")
        
        # Composantes connexes
        components = list(nx.connected_components(G))
        n_components = len(components)
        largest_component = max(components, key=len)
        largest_size = len(largest_component)
        
        results["components"] = {
            "n_components": n_components,
            "largest_component_size": largest_size,
            "largest_component_fraction": largest_size / G.number_of_nodes(),
            "isolated_nodes": sum(1 for node in G.nodes() if G.degree(node) == 0),
            "component_sizes": [len(c) for c in components]
        }
        
        print(f"\n🔗 Connectivité:")
        print(f"   - Nombre de composantes: {n_components}")
        print(f"   - Plus grande composante: {largest_size} nœuds ({results['components']['largest_component_fraction']*100:.2f}%)")
        print(f"   - Nœuds isolés: {results['components']['isolated_nodes']}")
        
        # Nœuds audio sans arêtes cross (orphelins potentiels)
        audio_with_cross = set()
        if self.cross_edges is not None:
            for i in range(self.n_cross_edges):
                audio_with_cross.add(self.cross_edges[0, i].item())
        
        orphan_audio = self.n_audio - len(audio_with_cross)
        results["orphan_audio"] = {
            "count": orphan_audio,
            "fraction": orphan_audio / self.n_audio if self.n_audio > 0 else 0
        }
        
        print(f"\n🎵 Nœuds audio sans arêtes cross (orphelins):")
        print(f"   - Nombre: {orphan_audio}")
        print(f"   - Fraction: {orphan_audio/self.n_audio*100:.2f}%")
        
        return results
    
    def analyze_audio_features(self) -> Dict[str, Any]:
        """
        Analyse les features audio.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DES FEATURES AUDIO")
        print("="*60)
        
        features = self.audio_features.numpy()
        
        results = {
            "shape": features.shape,
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "norm_mean": float(np.mean(np.linalg.norm(features, axis=1))),
            "norm_std": float(np.std(np.linalg.norm(features, axis=1))),
            "rank": int(np.linalg.matrix_rank(features))
        }
        
        print(f"\n📊 Statistiques des features audio:")
        print(f"   - Shape: {features.shape}")
        print(f"   - Moyenne: {results['mean']:.4f}")
        print(f"   - Écart-type: {results['std']:.4f}")
        print(f"   - Norme moyenne: {results['norm_mean']:.4f}")
        print(f"   - Rang de la matrice: {results['rank']}")
        
        return results
    
    def analyze_word_features(self) -> Dict[str, Any]:
        """
        Analyse les features word.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DES FEATURES WORD")
        print("="*60)
        
        features = self.word_features.numpy()
        
        results = {
            "shape": features.shape,
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "norm_mean": float(np.mean(np.linalg.norm(features, axis=1))),
            "norm_std": float(np.std(np.linalg.norm(features, axis=1))),
            "rank": int(np.linalg.matrix_rank(features))
        }
        
        print(f"\n📊 Statistiques des features word:")
        print(f"   - Shape: {features.shape}")
        print(f"   - Moyenne: {results['mean']:.4f}")
        print(f"   - Écart-type: {results['std']:.4f}")
        print(f"   - Norme moyenne: {results['norm_mean']:.4f}")
        print(f"   - Rang de la matrice: {results['rank']}")
        
        return results
    
    def analyze_cross_mapping(self) -> Dict[str, Any]:
        """
        Analyse le mapping audio-word.
        """
        print("\n" + "="*60)
        print("📊 ANALYSE DU MAPPING AUDIO-WORD")
        print("="*60)
        
        results = {}
        
        if self.cross_edges is not None and self.n_cross_edges > 0:
            # Nombre de mots par audio
            audio_words = defaultdict(int)
            word_audios = defaultdict(int)
            
            for i in range(self.n_cross_edges):
                audio_idx = self.cross_edges[0, i].item()
                word_idx = self.cross_edges[1, i].item()
                audio_words[audio_idx] += 1
                word_audios[word_idx] += 1
            
            audio_word_counts = list(audio_words.values())
            word_audio_counts = list(word_audios.values())
            
            results["audio_word_mapping"] = {
                "mean_words_per_audio": float(np.mean(audio_word_counts)),
                "std_words_per_audio": float(np.std(audio_word_counts)),
                "min_words_per_audio": int(min(audio_word_counts)),
                "max_words_per_audio": int(max(audio_word_counts)),
                "mean_audios_per_word": float(np.mean(word_audio_counts)),
                "std_audios_per_word": float(np.std(word_audio_counts)),
                "min_audios_per_word": int(min(word_audio_counts)),
                "max_audios_per_word": int(max(word_audio_counts)),
                "audio_with_multiple_words": sum(1 for c in audio_word_counts if c > 1),
                "word_with_multiple_audios": sum(1 for c in word_audio_counts if c > 1)
            }
            
            print(f"\n🔗 Mapping Audio → Word:")
            print(f"   - Moyenne mots par audio: {results['audio_word_mapping']['mean_words_per_audio']:.2f}")
            print(f"   - Min/Max mots par audio: {results['audio_word_mapping']['min_words_per_audio']} / {results['audio_word_mapping']['max_words_per_audio']}")
            print(f"   - Audios avec plusieurs mots: {results['audio_word_mapping']['audio_with_multiple_words']}")
            
            print(f"\n🔗 Mapping Word → Audio:")
            print(f"   - Moyenne audios par word: {results['audio_word_mapping']['mean_audios_per_word']:.2f}")
            print(f"   - Min/Max audios par word: {results['audio_word_mapping']['min_audios_per_word']} / {results['audio_word_mapping']['max_audios_per_word']}")
            print(f"   - Words avec plusieurs audios: {results['audio_word_mapping']['word_with_multiple_audios']}")
        
        return results
    
    def generate_plots(self) -> None:
        """
        Génère des visualisations du graphe.
        """
        print("\n" + "="*60)
        print("📊 GÉNÉRATION DES VISUALISATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribution des degrés audio-audio
        if self.audio_audio_edges is not None and self.n_audio_audio_edges > 0:
            degrees = torch.zeros(self.n_audio, dtype=torch.long)
            for i in range(self.n_audio_audio_edges):
                degrees[self.audio_audio_edges[0, i]] += 1
                degrees[self.audio_audio_edges[1, i]] += 1
            
            ax = axes[0, 0]
            ax.hist(degrees.numpy(), bins=20, alpha=0.7, color='blue')
            ax.set_title("Distribution des degrés Audio-Audio")
            ax.set_xlabel("Degré")
            ax.set_ylabel("Fréquence")
        
        # 2. Distribution des degrés cross
        if self.cross_edges is not None and self.n_cross_edges > 0:
            audio_degrees = torch.zeros(self.n_audio, dtype=torch.long)
            word_degrees = torch.zeros(self.n_word, dtype=torch.long)
            
            for i in range(self.n_cross_edges):
                audio_degrees[self.cross_edges[0, i]] += 1
                word_degrees[self.cross_edges[1, i]] += 1
            
            ax = axes[0, 1]
            ax.hist(audio_degrees.numpy(), bins=20, alpha=0.5, label="Audio", color='green')
            ax.hist(word_degrees.numpy(), bins=20, alpha=0.5, label="Word", color='red')
            ax.set_title("Distribution des degrés Cross")
            ax.set_xlabel("Degré")
            ax.set_ylabel("Fréquence")
            ax.legend()
        
        # 3. Distribution des poids
        if self.audio_audio_weights is not None and len(self.audio_audio_weights) > 0:
            ax = axes[0, 2]
            ax.hist(self.audio_audio_weights.numpy(), bins=20, alpha=0.7, color='blue')
            ax.set_title("Distribution des poids Audio-Audio")
            ax.set_xlabel("Poids")
            ax.set_ylabel("Fréquence")
        
        # 4. Distribution des poids cross
        if self.cross_weights is not None and len(self.cross_weights) > 0:
            ax = axes[1, 0]
            weights = self.cross_weights.numpy()
            ax.hist(weights, bins=20, alpha=0.7, color='green')
            ax.axvline(x=1.0, color='red', linestyle='--', label='Poids exact (1.0)')
            ax.set_title("Distribution des poids Cross")
            ax.set_xlabel("Poids")
            ax.set_ylabel("Fréquence")
            ax.legend()
        
        # 5. Matrice de similarité audio-audio (sample)
        if self.n_audio > 10:
            sample_size = min(100, self.n_audio)
            sample_features = self.audio_features[:sample_size].numpy()
            sample_sim = np.dot(sample_features, sample_features.T)
            
            ax = axes[1, 1]
            im = ax.imshow(sample_sim, cmap='viridis', aspect='auto')
            ax.set_title(f"Similarité Audio-Audio (sample {sample_size})")
            ax.set_xlabel("Nœud audio")
            ax.set_ylabel("Nœud audio")
            plt.colorbar(im, ax=ax)
        
        # 6. Distribution des normes
        ax = axes[1, 2]
        audio_norms = np.linalg.norm(self.audio_features.numpy(), axis=1)
        word_norms = np.linalg.norm(self.word_features.numpy(), axis=1)
        ax.hist(audio_norms, bins=20, alpha=0.5, label="Audio", color='blue')
        ax.hist(word_norms, bins=20, alpha=0.5, label="Word", color='red')
        ax.set_title("Distribution des normes des features")
        ax.set_xlabel("Norme")
        ax.set_ylabel("Fréquence")
        ax.legend()
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = self.output_dir / "graph_analysis_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualisations sauvegardées: {plot_path}")
        plt.close()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet d'analyse.
        """
        print("\n" + "="*60)
        print("📊 GÉNÉRATION DU RAPPORT COMPLET")
        print("="*60)
        
        report = {
            "structure": self.analyze_structure(),
            "degrees": self.analyze_degrees(),
            "weights": self.analyze_weights(),
            "connectivity": self.analyze_connectivity(),
            "audio_features": self.analyze_audio_features(),
            "word_features": self.analyze_word_features(),
            "cross_mapping": self.analyze_cross_mapping()
        }
        
        # Sauvegarder le rapport
        report_path = self.output_dir / "graph_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"   ✅ Rapport sauvegardé: {report_path}")
        
        # Sauvegarder en CSV pour analyse
        self._save_csv_summary(report)
        
        # Générer les visualisations
        self.generate_plots()
        
        # Générer un rapport texte
        self._generate_text_report(report)
        
        return report
    
    def _save_csv_summary(self, report: Dict[str, Any]):
        """Sauvegarde un résumé en CSV."""
        # Résumé des métriques principales
        summary = {
            "metric": [],
            "value": []
        }
        
        # Structure
        summary["metric"].extend(["n_audio", "n_word", "n_audio_audio_edges", "n_word_word_edges", "n_cross_edges"])
        summary["value"].extend([
            report["structure"]["nodes"]["audio"],
            report["structure"]["nodes"]["word"],
            report["structure"]["edges"]["audio_audio"],
            report["structure"]["edges"]["word_word"],
            report["structure"]["edges"]["cross"]
        ])
        
        # Densités
        summary["metric"].extend(["density_audio_audio", "density_word_word", "density_cross"])
        summary["value"].extend([
            report["structure"]["density"]["audio_audio"],
            report["structure"]["density"]["word_word"],
            report["structure"]["density"]["cross"]
        ])
        
        # Connectivité
        if "components" in report["connectivity"]:
            summary["metric"].append("n_components")
            summary["value"].append(report["connectivity"]["components"]["n_components"])
            summary["metric"].append("isolated_nodes")
            summary["value"].append(report["connectivity"]["components"]["isolated_nodes"])
            summary["metric"].append("orphan_audio")
            summary["value"].append(report["connectivity"]["orphan_audio"]["count"])
        
        # Poids
        if "cross_weights" in report["weights"]:
            summary["metric"].append("cross_weight_mean")
            summary["value"].append(report["weights"]["cross_weights"]["mean"])
            summary["metric"].append("cross_weight_exact_match_fraction")
            summary["value"].append(report["weights"]["cross_weights"]["exact_match_fraction"])
        
        df = pd.DataFrame(summary)
        csv_path = self.output_dir / "graph_analysis_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Résumé CSV sauvegardé: {csv_path}")
    
    def _generate_text_report(self, report: Dict[str, Any]):
        """Génère un rapport texte lisible."""
        report_path = self.output_dir / "graph_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ANALYSE DU GRAPHE HÉTÉROGÈNE\n")
            f.write("="*80 + "\n\n")
            
            # 1. Structure
            f.write("1. STRUCTURE DU GRAPHE\n")
            f.write("-"*40 + "\n")
            f.write(f"   Nœuds audio: {report['structure']['nodes']['audio']}\n")
            f.write(f"   Nœuds word: {report['structure']['nodes']['word']}\n")
            f.write(f"   Total nœuds: {report['structure']['nodes']['total']}\n")
            f.write(f"   Arêtes audio-audio: {report['structure']['edges']['audio_audio']}\n")
            f.write(f"   Arêtes word-word: {report['structure']['edges']['word_word']}\n")
            f.write(f"   Arêtes cross: {report['structure']['edges']['cross']}\n")
            f.write(f"   Total arêtes: {report['structure']['edges']['total']}\n")
            f.write(f"   Densité audio-audio: {report['structure']['density']['audio_audio']:.4f}\n")
            f.write(f"   Densité word-word: {report['structure']['density']['word_word']:.4f}\n")
            f.write(f"   Densité cross: {report['structure']['density']['cross']:.4f}\n\n")
            
            # 2. Connectivité
            if "components" in report["connectivity"]:
                f.write("2. CONNECTIVITÉ\n")
                f.write("-"*40 + "\n")
                f.write(f"   Composantes connexes: {report['connectivity']['components']['n_components']}\n")
                f.write(f"   Plus grande composante: {report['connectivity']['components']['largest_component_size']} nœuds ({report['connectivity']['components']['largest_component_fraction']*100:.2f}%)\n")
                f.write(f"   Nœuds isolés: {report['connectivity']['components']['isolated_nodes']}\n")
                f.write(f"   Nœuds audio orphelins: {report['connectivity']['orphan_audio']['count']} ({report['connectivity']['orphan_audio']['fraction']*100:.2f}%)\n\n")
            
            # 3. Mapping
            if "cross_mapping" in report and "audio_word_mapping" in report["cross_mapping"]:
                f.write("3. MAPPING AUDIO-WORD\n")
                f.write("-"*40 + "\n")
                f.write(f"   Moyenne mots par audio: {report['cross_mapping']['audio_word_mapping']['mean_words_per_audio']:.2f}\n")
                f.write(f"   Min/Max mots par audio: {report['cross_mapping']['audio_word_mapping']['min_words_per_audio']} / {report['cross_mapping']['audio_word_mapping']['max_words_per_audio']}\n")
                f.write(f"   Moyenne audios par word: {report['cross_mapping']['audio_word_mapping']['mean_audios_per_word']:.2f}\n")
                f.write(f"   Min/Max audios par word: {report['cross_mapping']['audio_word_mapping']['min_audios_per_word']} / {report['cross_mapping']['audio_word_mapping']['max_audios_per_word']}\n\n")
            
            # 4. Poids cross
            if "cross_weights" in report["weights"]:
                f.write("4. POIDS DES ARÊTES CROSS\n")
                f.write("-"*40 + "\n")
                f.write(f"   Moyenne: {report['weights']['cross_weights']['mean']:.4f}\n")
                f.write(f"   Écart-type: {report['weights']['cross_weights']['std']:.4f}\n")
                f.write(f"   Min/Max: {report['weights']['cross_weights']['min']:.4f} / {report['weights']['cross_weights']['max']:.4f}\n")
                f.write(f"   Correspondances exactes (poids=1.0): {report['weights']['cross_weights']['exact_match_count']} ({report['weights']['cross_weights']['exact_match_fraction']*100:.2f}%)\n")
                f.write(f"   Percentile 95%: {report['weights']['cross_weights']['percentiles']['95']:.4f}\n\n")
            
            # 5. Features
            f.write("5. FEATURES\n")
            f.write("-"*40 + "\n")
            f.write(f"   Audio - Shape: {report['audio_features']['shape']}\n")
            f.write(f"   Audio - Norme moyenne: {report['audio_features']['norm_mean']:.4f}\n")
            f.write(f"   Audio - Rang: {report['audio_features']['rank']}\n")
            f.write(f"   Word - Shape: {report['word_features']['shape']}\n")
            f.write(f"   Word - Norme moyenne: {report['word_features']['norm_mean']:.4f}\n")
            f.write(f"   Word - Rang: {report['word_features']['rank']}\n\n")
            
            # 6. Degrés
            if "audio_audio_degrees" in report["degrees"]:
                f.write("6. DEGRÉS\n")
                f.write("-"*40 + "\n")
                f.write(f"   Audio-Audio - Moyenne: {report['degrees']['audio_audio_degrees']['mean']:.2f}\n")
                f.write(f"   Audio-Audio - Min/Max: {report['degrees']['audio_audio_degrees']['min']} / {report['degrees']['audio_audio_degrees']['max']}\n")
            if "cross_degrees_audio" in report["degrees"]:
                f.write(f"   Cross (Audio) - Moyenne: {report['degrees']['cross_degrees_audio']['mean']:.2f}\n")
                f.write(f"   Cross (Audio) - Min/Max: {report['degrees']['cross_degrees_audio']['min']} / {report['degrees']['cross_degrees_audio']['max']}\n")
                f.write(f"   Cross (Word) - Moyenne: {report['degrees']['cross_degrees_word']['mean']:.2f}\n")
                f.write(f"   Cross (Word) - Min/Max: {report['degrees']['cross_degrees_word']['min']} / {report['degrees']['cross_degrees_word']['max']}\n")
        
        print(f"   ✅ Rapport texte sauvegardé: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse complète du graphe hétérogène")
    parser.add_argument("--graph_path", type=str, required=True, 
                       help="Chemin vers le fichier du graphe (.pt)")
    parser.add_argument("--config", type=str, default=None,
                       help="Fichier de configuration (optionnel)")
    parser.add_argument("--output_dir", type=str, default="./graph_analysis",
                       help="Dossier de sortie")
    args = parser.parse_args()
    
    # Vérifier que le graphe existe
    if not Path(args.graph_path).exists():
        print(f"❌ Fichier non trouvé: {args.graph_path}")
        return
    
    # Créer l'analyseur
    analyzer = GraphAnalyzer(
        graph_path=args.graph_path,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Générer le rapport
    report = analyzer.generate_report()
    
    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE")
    print("="*60)
    print(f"📁 Résultats sauvegardés dans: {args.output_dir}")
    print(f"   - graph_analysis_report.json")
    print(f"   - graph_analysis_report.txt")
    print(f"   - graph_analysis_summary.csv")
    print(f"   - graph_analysis_plots.png")


if __name__ == "__main__":
    main()
