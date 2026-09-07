"""
prepare_bajia.py
Script de prétraitement pour le dataset BAJIA.

Structure attendue:
data/raw/bajia/
├── audio/
│   ├── loc_001/                   # Dossier du locuteur 1
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── mapping.tsv            # Contient: audio_filename, key, sentence, attempts
│   ├── loc_002/
│   │   ├── audio1.wav
│   │   └── mapping.tsv
│   ├── loc_003/
│   └── loc_004/
│
└── (autres fichiers éventuels)

Structure de sortie:
data/processed/bajia/
├── metadata_train.csv             # Données d'entraînement
├── metadata_test.csv              # Données de test
└── dataset_stats.txt              # Statistiques du dataset
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle


class BajiaPreprocessor:
    """
    Prétraitement du dataset BAJIA.
    Sépare les données en train (80%) et test (20%) par locuteur.
    Les chemins audio sont sauvegardés en ABSOLU.
    """
    
    def __init__(
        self,
        raw_dir: str = "./data/raw/bajia",
        output_dir: str = "./data/processed/bajia",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        use_absolute_paths: bool = True
    ):
        """
        Args:
            raw_dir: Dossier contenant les données brutes
            output_dir: Dossier de sortie pour les données traitées
            train_ratio: Proportion des données pour l'entraînement (0-1)
            random_seed: Graine aléatoire pour la reproductibilité
            use_absolute_paths: Si True, utilise des chemins absolus dans metadata.csv
        """
        self.raw_dir = Path(raw_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.use_absolute_paths = use_absolute_paths
        
        # Initialiser les conteneurs
        self.all_examples = []
        
        # Vérifier que le dossier raw existe
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Dossier raw non trouvé: {self.raw_dir}")
        
        print(f"📁 Prétraitement du dataset BAJIA")
        print(f"   Source: {self.raw_dir}")
        print(f"   Destination: {self.output_dir}")
        print(f"   Ratio train/test: {train_ratio:.0%}/{1-train_ratio:.0%}")
        print(f"   Chemins absolus: {'Oui' if use_absolute_paths else 'Non'}")
    
    def load_mapping(self, mapping_path: Path) -> Dict[str, Dict[str, str]]:
        """
        Charge un fichier mapping.tsv.
        
        Format attendu (TSV avec colonnes):
        audio_filename    key    sentence    attempts
        
        Returns:
            Dict {audio_filename: {'key': key, 'sentence': sentence, 'attempts': attempts}}
        """
        if not mapping_path.exists():
            raise FileNotFoundError(f"Fichier mapping non trouvé: {mapping_path}")
        
        mapping = {}
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Vérifier les colonnes requises
            required_cols = ['audio_filename', 'sentence']
            for col in required_cols:
                if col not in reader.fieldnames:
                    print(f"   ⚠️ Colonne '{col}' non trouvée. Colonnes disponibles: {reader.fieldnames}")
            
            for row in reader:
                audio_filename = row.get('audio_filename', '').strip()
                if not audio_filename:
                    continue
                
                mapping[audio_filename] = {
                    'key': row.get('key', '').strip(),
                    'sentence': row.get('sentence', '').strip(),
                    'attempts': row.get('attempts', '').strip()
                }
        
        print(f"   ✅ {len(mapping)} entrées chargées depuis {mapping_path.name}")
        return mapping
    
    def scan_audio_files(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Parcourt tous les sous-dossiers audio pour trouver les fichiers.
        
        Returns:
            Dict {speaker_id: [(absolute_audio_path, audio_filename, sentence), ...]}
        """
        audio_dir = self.raw_dir / "audio"
        
        if not audio_dir.exists():
            raise FileNotFoundError(f"Dossier audio non trouvé: {audio_dir}")
        
        speaker_files = defaultdict(list)
        
        # Trouver tous les sous-dossiers dans audio/
        subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            # Si pas de sous-dossiers, chercher directement les fichiers wav
            wav_files = list(audio_dir.glob('*.wav'))
            if wav_files:
                # Chercher un mapping.tsv à la racine
                mapping_path = audio_dir / "mapping.tsv"
                if mapping_path.exists():
                    mapping = self.load_mapping(mapping_path)
                    for wav_path in wav_files:
                        abs_path = wav_path.resolve()
                        filename = wav_path.name
                        if filename in mapping:
                            sentence = mapping[filename]['sentence']
                            speaker_files['unknown'].append((str(abs_path), filename, sentence))
                else:
                    print(f"   ⚠️ Aucun mapping.tsv trouvé dans {audio_dir}")
                    for wav_path in wav_files:
                        abs_path = wav_path.resolve()
                        speaker_files['unknown'].append((str(abs_path), wav_path.name, ""))
                print(f"   Aucun sous-dossier trouvé, mais {len(wav_files)} fichiers .wav trouvés")
                print(f"   Assignés au locuteur 'unknown'")
            return speaker_files
        
        for subdir in sorted(subdirs):
            speaker_id = subdir.name
            
            # Chercher le mapping.tsv dans ce dossier
            mapping_path = subdir / "mapping.tsv"
            if mapping_path.exists():
                mapping = self.load_mapping(mapping_path)
            else:
                print(f"   ⚠️ mapping.tsv non trouvé dans {subdir}")
                mapping = {}
            
            # Trouver tous les fichiers audio
            wav_files = list(subdir.glob('*.wav'))
            for wav_path in wav_files:
                abs_path = wav_path.resolve()
                filename = wav_path.name
                
                # Récupérer la transcription depuis le mapping
                if filename in mapping:
                    sentence = mapping[filename]['sentence']
                else:
                    sentence = ""
                    print(f"      ⚠️ {filename} non trouvé dans mapping.tsv")
                
                speaker_files[speaker_id].append((str(abs_path), filename, sentence))
            
            print(f"   Locuteur {speaker_id}: {len(speaker_files[speaker_id])} fichiers")
        
        return speaker_files
    
    def create_metadata(
        self,
        speaker_files: Dict[str, List[Tuple[str, str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Crée les métadonnées à partir des fichiers audio.
        
        Returns:
            List de dictionnaires avec les colonnes: audio_path, speaker_id, split, transcription, duration, file_stem
        """
        metadata = []
        
        for speaker_id, files in speaker_files.items():
            for audio_path, filename, sentence in files:
                # Vérifier si on utilise des chemins absolus ou relatifs
                if self.use_absolute_paths:
                    final_path = audio_path
                else:
                    final_path = os.path.relpath(audio_path, os.getcwd())
                
                # Extraire le file_stem (sans extension)
                file_stem = Path(filename).stem
                
                metadata.append({
                    'audio_path': final_path,
                    'speaker_id': speaker_id,
                    'split': 'train',  # Sera attribué plus tard
                    'transcription': sentence,
                    'duration': 0.000,  # Valeur par défaut, à calculer si nécessaire
                    'file_stem': file_stem
                })
        
        print(f"   ✅ {len(metadata)} fichiers audio traités")
        
        # Afficher un exemple de chemin
        if metadata:
            print(f"   Exemple de chemin audio: {metadata[0]['audio_path']}")
        
        return metadata
    
    def split_data(self, metadata: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Sépare les données en train et test par locuteur.
        
        Returns:
            Tuple (train_data, test_data)
        """
        # Grouper par locuteur
        speaker_examples = defaultdict(list)
        for exp in metadata:
            speaker_examples[exp['speaker_id']].append(exp)
        
        train_data = []
        test_data = []
        
        # Fixer la graine aléatoire
        random.seed(self.random_seed)
        
        for speaker_id, examples in speaker_examples.items():
            # Mélanger les exemples
            shuffled = examples.copy()
            random.shuffle(shuffled)
            
            # Calculer le point de séparation
            split_idx = int(len(shuffled) * self.train_ratio)
            
            # Assigner les splits
            train_examples = shuffled[:split_idx]
            test_examples = shuffled[split_idx:]
            
            # Mettre à jour le champ 'split'
            for exp in train_examples:
                exp['split'] = 'train'
            for exp in test_examples:
                exp['split'] = 'test'
            
            train_data.extend(train_examples)
            test_data.extend(test_examples)
            
            print(f"   Locuteur {speaker_id}: {len(train_examples)} train / {len(test_examples)} test")
        
        return train_data, test_data
    
    def save_statistics(self, train_data: List[Dict], test_data: List[Dict]):
        """
        Sauvegarde les statistiques du dataset.
        """
        all_data = train_data + test_data
        
        # Compter les locuteurs
        speakers = set(exp['speaker_id'] for exp in all_data)
        
        # Compter par locuteur
        speaker_counts = defaultdict(int)
        for exp in all_data:
            speaker_counts[exp['speaker_id']] += 1
        
        # Compter les splits
        split_counts = defaultdict(int)
        for exp in all_data:
            split_counts[exp['split']] += 1
        
        # Générer le rapport
        stats_path = self.output_dir / 'dataset_stats.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES DU DATASET BAJIA\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Chemin absolu du dataset: {self.output_dir}\n")
            f.write(f"Chemin absolu des fichiers: {self.raw_dir}\n\n")
            
            f.write(f"Nombre total d'exemples: {len(all_data)}\n")
            f.write(f"Nombre de locuteurs: {len(speakers)}\n\n")
            
            f.write("Répartition par split:\n")
            f.write(f"  - Train: {split_counts['train']} ({split_counts['train']/len(all_data):.1%})\n")
            f.write(f"  - Test: {split_counts['test']} ({split_counts['test']/len(all_data):.1%})\n\n")
            
            f.write("Répartition par locuteur:\n")
            for speaker, count in sorted(speaker_counts.items()):
                train_count = sum(1 for e in train_data if e['speaker_id'] == speaker)
                test_count = sum(1 for e in test_data if e['speaker_id'] == speaker)
                f.write(f"  - {speaker}: {count} exemples (train: {train_count}, test: {test_count})\n")
            
            f.write("\nExemples de chemins (train):\n")
            for i, exp in enumerate(train_data[:3]):
                f.write(f"  {i+1}. {exp['audio_path']}\n")
            if len(train_data) > 3:
                f.write(f"  ... et {len(train_data) - 3} autres\n")
            
            f.write("\nExemples de chemins (test):\n")
            for i, exp in enumerate(test_data[:3]):
                f.write(f"  {i+1}. {exp['audio_path']}\n")
            if len(test_data) > 3:
                f.write(f"  ... et {len(test_data) - 3} autres\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   ✅ Statistiques sauvegardées: {stats_path}")
        
        # Sauvegarder les infos en pickle
        stats_pkl_path = self.output_dir / 'speakers_info.pkl'
        with open(stats_pkl_path, 'wb') as f:
            pickle.dump({
                'speakers': list(speakers),
                'speaker_counts': dict(speaker_counts),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'total_examples': len(all_data),
                'num_speakers': len(speakers),
                'raw_dir': str(self.raw_dir),
                'output_dir': str(self.output_dir),
                'train_ratio': self.train_ratio,
                'random_seed': self.random_seed,
                'use_absolute_paths': self.use_absolute_paths
            }, f)
        print(f"   ✅ Infos locuteurs sauvegardées: {stats_pkl_path}")
    
    def run(self):
        """
        Exécute tout le pipeline de prétraitement.
        """
        print("\n" + "="*60)
        print("PRÉTRAITEMENT DU DATASET BAJIA")
        print("="*60)
        
        # 1. Scanner les fichiers audio
        print("\n🎵 Étape 1: Scan des fichiers audio")
        speaker_files = self.scan_audio_files()
        
        if not speaker_files:
            print("❌ Aucun fichier audio trouvé!")
            return
        
        # 2. Créer les métadonnées
        print("\n📊 Étape 2: Création des métadonnées")
        metadata = self.create_metadata(speaker_files)
        
        if not metadata:
            print("❌ Aucune donnée valide trouvée!")
            return
        
        # 3. Split train/test
        print("\n📊 Étape 3: Split train/test")
        train_data, test_data = self.split_data(metadata)
        
        if not train_data or not test_data:
            print("❌ Split impossible!")
            return
        
        # 4. Sauvegarder les métadonnées
        print("\n💾 Étape 4: Sauvegarde des métadonnées")
        
        # Sauvegarder train
        train_path = self.output_dir / 'metadata_train.csv'
        with open(train_path, 'w', newline='', encoding='utf-8') as f:
            if train_data:
                fieldnames = train_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(train_data)
        
        # Sauvegarder test
        test_path = self.output_dir / 'metadata_test.csv'
        with open(test_path, 'w', newline='', encoding='utf-8') as f:
            if test_data:
                fieldnames = test_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(test_data)
        
        print(f"   ✅ Métadonnées sauvegardées:")
        print(f"      - Train: {train_path} ({len(train_data)} exemples)")
        print(f"      - Test: {test_path} ({len(test_data)} exemples)")
        print(f"      - Chemins: {'Absolus' if self.use_absolute_paths else 'Relatifs'}")
        
        # 5. Sauvegarder les statistiques
        print("\n📈 Étape 5: Sauvegarde des statistiques")
        self.save_statistics(train_data, test_data)
        
        print("\n" + "="*60)
        print("✅ PRÉTRAITEMENT TERMINÉ!")
        print("="*60)
        print(f"\n📁 Données sauvegardées dans: {self.output_dir}")
        print(f"   - metadata_train.csv: {len(train_data)} exemples")
        print(f"   - metadata_test.csv: {len(test_data)} exemples")
        print(f"   - dataset_stats.txt: statistiques détaillées")
        print(f"   - speakers_info.pkl: informations sur les locuteurs")
        print(f"\n📂 Les chemins audio sont en {'ABSOLU' if self.use_absolute_paths else 'RELATIF'}")


def main():
    """Script principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prétraitement du dataset BAJIA")
    parser.add_argument("--raw_dir", type=str, default="data/raw/badja",
                       help="Dossier contenant les données brutes")
    parser.add_argument("--output_dir", type=str, default="data/processed/bajia",
                       help="Dossier de sortie")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Proportion pour l'entraînement (0-1)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Graine aléatoire pour la reproductibilité")
    parser.add_argument("--relative_paths", action="store_true",
                       help="Utiliser des chemins relatifs (par défaut: absolus)")
    
    args = parser.parse_args()
    
    preprocessor = BajiaPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        use_absolute_paths=not args.relative_paths
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
