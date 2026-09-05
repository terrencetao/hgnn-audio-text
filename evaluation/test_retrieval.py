"""
test_retrieval.py

Script de test pour le retrieval avec les 3 options d'inférence.
Utilise les méthodes de RetrievalEvaluator.
Prend un CSV de test en entrée et retourne 2 CSV :
1. audio_retrieval_results.csv : résumé par audio avec les textes trouvés
2. text_retrieval_results.csv : résumé par texte avec les audios trouvés
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import List, Dict, Tuple
import json
from datetime import datetime

from retrieval import RetrievalEvaluator, extract_reference_graph
from graph.similarity import LinguisticRepresentation


class RetrievalTester:
    """
    Testeur de retrieval pour les 3 options d'inférence.
    Utilise directement les méthodes de RetrievalEvaluator.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        representation_type: str = "labse",
        max_edges_per_node: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.representation_type = LinguisticRepresentation(representation_type)
        self.max_edges_per_node = max_edges_per_node
        
        # Initialiser l'évaluateur (il gère tout : config, modèles, graphe)
        self.evaluator = RetrievalEvaluator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            representation_type=self.representation_type,
            max_edges_per_node=max_edges_per_node,
            device=device
        )
        
        # Récupérer les noms des modèles
        self.backbone_name = self.evaluator.backbone_model_name
        self.rep_name = self.evaluator.representation_model_name
        self.threshold = self.evaluator.similarity_threshold
        
        print(f"\n🔧 Testeur retrieval initialisé:")
        print(f"   - Backbone: {self.backbone_name}")
        print(f"   - Représentation: {self.rep_name}")
        print(f"   - Seuil de similarité: {self.threshold}")
        print(f"   - Max arêtes par nœud: {self.max_edges_per_node}")
    
    def test_on_csv(
        self,
        test_csv_path: str,
        reference_fraction: float = 0.5,
        k_values: List[int] = [1, 5, 10],
        confidence_threshold: float = 0.7,
        top_k_candidates: int = 3,
        seed: int = 42,
        output_dir: str = "./test_results"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Teste le retrieval sur un CSV de test.
        
        Args:
            test_csv_path: Chemin vers le CSV de test (audio_path, transcription)
            reference_fraction: Fraction du graphe à utiliser comme référence
            k_values: Valeurs de k pour le retrieval
            confidence_threshold: Seuil de confiance pour l'option 3
            top_k_candidates: Nombre de candidats pour l'option 3
            seed: Graine aléatoire
            output_dir: Dossier de sortie
        
        Returns:
            audio_results_df: DataFrame avec résultats par audio
            text_results_df: DataFrame avec résultats par texte
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Charger les données de test
        print(f"\n📂 Chargement du CSV de test: {test_csv_path}")
        test_df = pd.read_csv(test_csv_path)
        test_transcriptions = test_df['transcription'].tolist()
        
        # Créer un mapping transcription -> index
        text_to_idx = {text: i for i, text in enumerate(test_transcriptions)}
        idx_to_text = {i: text for i, text in enumerate(test_transcriptions)}
        
        # 2. Créer le graphe de référence
        print(f"\n🔨 Construction du graphe de référence (fraction={reference_fraction})...")
        reference_graph = extract_reference_graph(
            self.evaluator.full_graph,
            reference_fraction=reference_fraction,
            seed=seed
        )
        
        # 3. Encoder les transcriptions de test
        print(f"\n🔤 Encodage des transcriptions de test ({len(test_df)} samples)...")
        
        # Préparer les données pour l'encodage linguistique
        if self.representation_type in [LinguisticRepresentation.BAG_OF_PHONEMES, 
                                        LinguisticRepresentation.ARTICULATORY]:
            phrases_phonemes = self.evaluator.get_phoneme_segmentation(test_transcriptions)
        else:
            phrases_phonemes = None
        
        text_embeddings = self.evaluator.encode_transcriptions(
            test_transcriptions, 
            phrases_phonemes
        )
        
        # 4. Traiter chaque audio de test
        print(f"\n🎵 Traitement des audios de test ({len(test_df)} samples)...")
        
        # Résultats par audio
        audio_results = []
        
        # Résultats par texte
        text_retrieval_results = {text: [] for text in test_transcriptions}
        
        # Statistiques
        stats = {
            "option1": {"correct": 0, "total": 0, "top_k": {k: 0 for k in k_values}},
            "option2": {"correct": 0, "total": 0, "top_k": {k: 0 for k in k_values}},
            "option3": {"correct": 0, "total": 0, "top_k": {k: 0 for k in k_values}}
        }
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Traitement"):
            audio_path = Path(row['audio_path'])
            correct_transcription = row['transcription']
            correct_idx = text_to_idx[correct_transcription]
            
            # Extraire les features audio
            try:
                audio_features = self.evaluator.builder.extract_audio_features_from_path(
                    str(audio_path)
                )
            except Exception as e:
                print(f"❌ Erreur sur {audio_path.name}: {e}")
                continue
            
            # Résultat pour cet audio
            audio_result = {
                "audio_path": str(audio_path),
                "audio_name": audio_path.name,
                "correct_transcription": correct_transcription,
            }
            
            print(f'audio features : {audio_features.shape}')
            # Tester les 3 options
            for option_key in ["option1", "option2", "option3"]:
                option_names = {
                    "option1": "Option 1 (Audio Only)",
                    "option2": "Option 2 (Hétérogène)",
                    "option3": "Option 3 (Reconnection)"
                }
                
                # Utiliser la méthode infer_audio de l'évaluateur
                embedding = self.evaluator.infer_audio(
                    audio_features,
                    reference_graph,
                    option=option_key,
                    confidence_threshold=confidence_threshold,
                    top_k_candidates=top_k_candidates
                )
                
                # Faire le retrieval avec la méthode de l'évaluateur
                found_texts = []
                found_indices = []
                found_scores = []
                
                for k in k_values:
                    indices, scores = self.evaluator.retrieve(
                        embedding, text_embeddings, k=k
                    )
                    
                    texts = [idx_to_text[i] for i in indices]
                    is_correct = correct_idx in indices
                    
                    # Stocker pour le top k
                    audio_result[f"{option_key}_top{k}_texts"] = "|".join(texts)
                    audio_result[f"{option_key}_top{k}_scores"] = "|".join([f"{s:.4f}" for s in scores])
                    audio_result[f"{option_key}_top{k}_correct"] = is_correct
                    
                    # Mettre à jour les stats
                    stats[option_key]["top_k"][k] += 1 if is_correct else 0
                    
                    # Pour le plus petit k, stocker les textes trouvés
                    if k == min(k_values):
                        found_texts = texts
                        found_indices = indices
                        found_scores = scores
                
                # Vérifier si trouvé
                is_found = correct_transcription in found_texts
                stats[option_key]["total"] += 1
                if is_found:
                    stats[option_key]["correct"] += 1
                
                audio_result[f"{option_key}_found"] = is_found
                audio_result[f"{option_key}_found_texts"] = "|".join(found_texts)
                audio_result[f"{option_key}_found_scores"] = "|".join([f"{s:.4f}" for s in found_scores])
                audio_result[f"{option_key}_confidence"] = found_scores[0] if found_scores else 0.0
                
                # Ajouter aux résultats par texte
                for text, score in zip(found_texts, found_scores):
                    text_retrieval_results[text].append({
                        "audio_path": str(audio_path),
                        "audio_name": audio_path.name,
                        "option": option_names[option_key],
                        "score": float(score),
                        "is_correct": text == correct_transcription
                    })
            
            audio_results.append(audio_result)
        
        # 5. Calculer les métriques globales
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DES RÉSULTATS")
        print("="*60)
        
        summary_data = []
        for option_key in ["option1", "option2", "option3"]:
            option_names = {
                "option1": "Option 1 (Audio Only)",
                "option2": "Option 2 (Hétérogène)",
                "option3": "Option 3 (Reconnection)"
            }
            option_name = option_names[option_key]
            
            total = stats[option_key]["total"]
            correct = stats[option_key]["correct"]
            accuracy = correct / total if total > 0 else 0.0
            
            print(f"\n{option_name}:")
            print(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")
            for k in k_values:
                top_k_correct = stats[option_key]["top_k"][k]
                top_k_acc = top_k_correct / total if total > 0 else 0.0
                print(f"   Recall@{k}: {top_k_acc:.4f} ({top_k_correct}/{total})")
            
            summary_data.append({
                "option": option_name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                **{f"recall@{k}": stats[option_key]["top_k"][k] / total if total > 0 else 0.0 
                   for k in k_values}
            })
        
        # 6. Créer les DataFrames
        print("\n📁 Création des CSV de résultats...")
        
        # 6a. CSV des résultats par audio
        audio_results_df = pd.DataFrame(audio_results)
        
        # 6b. CSV des résultats par texte
        text_rows = []
        for text, entries in text_retrieval_results.items():
            for entry in entries:
                text_rows.append({
                    "transcription": text,
                    "audio_path": entry["audio_path"],
                    "audio_name": entry["audio_name"],
                    "option": entry["option"],
                    "score": entry["score"],
                    "is_correct": entry["is_correct"]
                })
        
        text_results_df = pd.DataFrame(text_rows)
        
        # 7. Sauvegarder les CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.backbone_name}_{self.rep_name}_f{reference_fraction}"
        
        # Audio results
        audio_csv_path = output_dir / f"audio_retrieval_results_{exp_name}_{timestamp}.csv"
        audio_results_df.to_csv(audio_csv_path, index=False, encoding='utf-8')
        print(f"   ✅ Audio results: {audio_csv_path}")
        
        # Text results
        text_csv_path = output_dir / f"text_retrieval_results_{exp_name}_{timestamp}.csv"
        text_results_df.to_csv(text_csv_path, index=False, encoding='utf-8')
        print(f"   ✅ Text results: {text_csv_path}")
        
        # Summary
        summary_csv_path = output_dir / f"summary_retrieval_results_{exp_name}_{timestamp}.csv"
        pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"   ✅ Summary: {summary_csv_path}")
        
        # JSON des métriques
        metrics_json = {
            "timestamp": timestamp,
            "experiment": exp_name,
            "config": {
                "reference_fraction": reference_fraction,
                "k_values": k_values,
                "confidence_threshold": confidence_threshold,
                "top_k_candidates": top_k_candidates,
                "seed": seed,
                "max_edges_per_node": self.max_edges_per_node,
                "similarity_threshold": self.threshold,
                "backbone": self.backbone_name,
                "representation": self.rep_name
            },
            "statistics": {
                option_key: {
                    "total": stats[option_key]["total"],
                    "correct": stats[option_key]["correct"],
                    "accuracy": stats[option_key]["correct"] / stats[option_key]["total"] if stats[option_key]["total"] > 0 else 0.0,
                    **{f"recall@{k}": stats[option_key]["top_k"][k] / stats[option_key]["total"] if stats[option_key]["total"] > 0 else 0.0
                       for k in k_values}
                }
                for option_key in ["option1", "option2", "option3"]
            }
        }
        
        metrics_path = output_dir / f"retrieval_metrics_{exp_name}_{timestamp}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, default=str)
        print(f"   ✅ Metrics: {metrics_path}")
        
        print("\n" + "="*60)
        print("✅ TEST TERMINÉ")
        print("="*60)
        print(f"📁 Résultats sauvegardés dans: {output_dir}")
        
        return audio_results_df, text_results_df


def main():
    parser = argparse.ArgumentParser(description="Test retrieval sur CSV")
    parser.add_argument("--config", type=str, required=True, help="Fichier de configuration")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint du modèle")
    parser.add_argument("--test_csv", type=str, required=True, help="CSV de test (audio_path,transcription)")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Dossier de sortie")
    
    # Paramètres du test
    parser.add_argument("--reference_fraction", type=float, default=0.5, help="Fraction de référence")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10], help="Valeurs de k")
    parser.add_argument("--representation_type", type=str, default="labse",
                       choices=["labse", "bag_of_phonemes", "articulatory"],
                       help="Type de représentation linguistique")
    parser.add_argument("--max_edges_per_node", type=int, default=10, help="Max arêtes par nœud")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Seuil de confiance Option 3")
    parser.add_argument("--top_k_candidates", type=int, default=3, help="Candidats Option 3")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    
    args = parser.parse_args()
    
    # Vérifier que le CSV existe
    if not Path(args.test_csv).exists():
        print(f"❌ Fichier CSV non trouvé: {args.test_csv}")
        return
    
    # Initialiser le testeur
    tester = RetrievalTester(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        representation_type=args.representation_type,
        max_edges_per_node=args.max_edges_per_node
    )
    
    # Lancer le test
    audio_results, text_results = tester.test_on_csv(
        test_csv_path=args.test_csv,
        reference_fraction=args.reference_fraction,
        k_values=args.k_values,
        confidence_threshold=args.confidence_threshold,
        top_k_candidates=args.top_k_candidates,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n📊 Aperçu des résultats audio (top 5):")
    cols = ['audio_name', 'correct_transcription', 
            'option2_found', 'option2_found_texts']
    if all(c in audio_results.columns for c in cols):
        print(audio_results[cols].head(5))
    
    print("\n📊 Aperçu des résultats textes (top 10):")
    print(text_results.head(10))


if __name__ == "__main__":
    main()
