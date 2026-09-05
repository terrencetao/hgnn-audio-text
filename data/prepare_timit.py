"""
prepare_timit.py
Script de prétraitement pour le dataset TIMIT.

Structure attendue:
data/raw/timit/
├── TRAIN/
│   ├── DR1/
│   │   ├── FCJF0/
│   │   │   ├── sa1.wav
│   │   │   ├── sa1.PHN
│   │   │   ├── sa1.WRD
│   │   │   └── sa1.TXT
│   │   └── ...
│   └── ...
└── TEST/
    └── ...

Structure de sortie:
data/processed/timit/
├── metadata_train.csv
├── metadata_test.csv
├── metadata_full.csv
├── dataset_stats.txt
└── speakers_info.pkl
"""

import os
import re
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import wave
import struct
import numpy as np


class TIMITPreprocessor:
    """
    Prétraitement du dataset TIMIT.
    """
    
    def __init__(
        self,
        raw_dir: str = "./data/raw/timit",
        output_dir: str = "./data/processed/timit",
        use_absolute_paths: bool = True
    ):
        """
        Args:
            raw_dir: Dossier contenant les données brutes TIMIT
            output_dir: Dossier de sortie pour les données traitées
            use_absolute_paths: Si True, utilise des chemins absolus
        """
        self.raw_dir = Path(raw_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_absolute_paths = use_absolute_paths
        
        self.examples = []
        self.speakers = set()
        self.transcriptions = set()
        self.speaker_to_id = {}
        self.transcription_to_id = {}
        
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Dossier raw non trouvé: {self.raw_dir}")
        
        print(f"📁 Prétraitement du dataset TIMIT")
        print(f"   Source: {self.raw_dir}")
        print(f"   Destination: {self.output_dir}")
        print(f"   Chemins absolus: {'Oui' if use_absolute_paths else 'Non'}")
    
    def parse_phn_file(self, phn_path: Path) -> List[Tuple[int, int, str]]:
        """
        Parse un fichier .PHN (phonèmes).
        
        Format: start_time end_time phoneme
        Exemple: 0 2231 h#
        
        Returns:
            Liste de (start, end, phoneme)
        """
        phonemes = []
        with open(phn_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, phoneme = parts
                    phonemes.append((int(start), int(end), phoneme))
        return phonemes
    
    def parse_wrd_file(self, wrd_path: Path) -> List[Tuple[int, int, str]]:
        """
        Parse un fichier .WRD (mots).
        
        Format: start_time end_time word
        Exemple: 2260 7895 critical
        
        Returns:
            Liste de (start, end, word)
        """
        words = []
        with open(wrd_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    words.append((int(start), int(end), word))
        return words
    
    def parse_txt_file(self, txt_path: Path) -> str:
        """
        Parse un fichier .TXT (transcription complète).
        
        Format: start_time end_time sentence
        Exemple: 0 36250 Critical equipment needs proper maintenance.
        
        Returns:
            Transcription complète
        """
        with open(txt_path, 'r') as f:
            content = f.read().strip()
            # Extraire la transcription (après les timings)
            parts = content.split()
            if len(parts) >= 3:
                # Rejoindre tous les mots après les timings
                return ' '.join(parts[2:])
        return content
    
    def find_speaker_id(self, path: Path) -> str:
        """
        Extrait l'ID du locuteur du chemin.
        Exemple: TRAIN/DR1/FCJF0/ → FCJF0
        """
        return path.parent.name
    
    def find_dialect_region(self, path: Path) -> str:
        """
        Extrait la région dialectale du chemin.
        Exemple: TRAIN/DR1/ → DR1
        """
        return path.parent.parent.name
    
    def scan_directory(self, split: str = "TRAIN") -> List[Dict]:
        """
        Scanne un répertoire (TRAIN ou TEST) pour trouver tous les fichiers audio.
        
        Returns:
            Liste de dictionnaires avec les métadonnées
        """
        split_dir = self.raw_dir / split
        if not split_dir.exists():
            print(f"   ⚠️ Dossier {split} non trouvé: {split_dir}")
            return []
        
        examples = []
        
        # Parcourir tous les sous-dossiers
        for speaker_dir in split_dir.rglob('*'):
            if not speaker_dir.is_dir():
                continue
            
            # Vérifier s'il y a des fichiers .wav
            wav_files = list(speaker_dir.glob('*.WAV'))
            if not wav_files:
                continue
            
            speaker_id = speaker_dir.name
            dialect = speaker_dir.parent.name
            
            for wav_path in wav_files:
                # Fichiers associés
                stem = wav_path.stem
                phn_path = wav_path.with_suffix('.PHN')
                wrd_path = wav_path.with_suffix('.WRD')
                txt_path = wav_path.with_suffix('.TXT')
                
                # Lire les transcriptions
                if phn_path.exists():
                    phonemes = self.parse_phn_file(phn_path)
                else:
                    phonemes = []
                    print(f"   ⚠️ Fichier PHN manquant: {phn_path}")
                
                if wrd_path.exists():
                    words = self.parse_wrd_file(wrd_path)
                else:
                    words = []
                    print(f"   ⚠️ Fichier WRD manquant: {wrd_path}")
                
                if txt_path.exists():
                    full_transcription = self.parse_txt_file(txt_path)
                else:
                    full_transcription = ' '.join([w for _, _, w in words])
                    print(f"   ⚠️ Fichier TXT manquant: {txt_path}")
                
                # Chemin absolu ou relatif
                if self.use_absolute_paths:
                    audio_path = str(wav_path.resolve())
                else:
                    audio_path = str(wav_path.relative_to(self.raw_dir.parent.parent))
                
                examples.append({
                    'audio_path': audio_path,
                    'speaker_id': speaker_id,
                    'dialect_region': dialect,
                    'split': split,
                    'transcription': full_transcription,
                    'words': words,
                    'phonemes': phonemes,
                    'num_phonemes': len(phonemes),
                    'num_words': len(words),
                    'duration': self._get_audio_duration(wav_path),
                    'file_stem': stem
                })
                
                self.speakers.add(speaker_id)
                if full_transcription:
                    self.transcriptions.add(full_transcription)
        
        print(f"   {split}: {len(examples)} exemples trouvés")
        return examples
    
    def _get_audio_duration(self, wav_path: Path) -> float:
        """Calcule la durée d'un fichier WAV en secondes."""
        try:
            with wave.open(str(wav_path), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0
    
    def create_metadata(self):
        """Crée les métadonnées pour TRAIN et TEST."""
        print("\n📊 Étape 1: Scan des dossiers")
        
        train_examples = self.scan_directory("TRAIN")
        test_examples = self.scan_directory("TEST")
        
        all_examples = train_examples + test_examples
        print(f"   Total: {len(all_examples)} exemples")
        
        if not all_examples:
            print("❌ Aucun exemple trouvé!")
            return
        
        # Créer les mappings
        self.speaker_to_id = {s: i for i, s in enumerate(sorted(self.speakers))}
        self.transcription_to_id = {t: i for i, t in enumerate(sorted(self.transcriptions))}
        
        print(f"\n📝 Étape 2: Création des mappings")
        print(f"   - {len(self.speakers)} locuteurs uniques")
        print(f"   - {len(self.transcriptions)} transcriptions uniques")
        
        # Sauvegarder les métadonnées
        print(f"\n💾 Étape 3: Sauvegarde des métadonnées")
        self._save_metadata(train_examples, 'train')
        self._save_metadata(test_examples, 'test')
        self._save_metadata(all_examples, 'full')
        
        # Statistiques
        self._save_statistics(train_examples, test_examples, all_examples)
        
        print(f"\n✅ Prétraitement terminé!")
        print(f"📁 Dossier de sortie: {self.output_dir}")
    
    def _save_metadata(self, examples: List[Dict], name: str):
        """Sauvegarde les métadonnées dans un CSV."""
        if not examples:
            print(f"   ⚠️ Pas d'exemples pour {name}")
            return
        
        metadata_path = self.output_dir / f'metadata_{name}.csv'
        
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'audio_path', 'speaker_id', 'dialect_region', 'split',
                'transcription', 'num_phonemes', 'num_words',
                'duration', 'file_stem'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for ex in examples:
                # Nettoyer la transcription (enlever les caractères spéciaux)
                trans = ex['transcription']
                trans = re.sub(r'[^a-zA-Z0-9\s\']', '', trans)
                ex['full_transcription'] = trans
                
                writer.writerow({
                    'audio_path': ex['audio_path'],
                    'speaker_id': ex['speaker_id'],
                    'dialect_region': ex['dialect_region'],
                    'split': ex['split'],
                    'transcription': ex['transcription'],
                    'num_phonemes': ex['num_phonemes'],
                    'num_words': ex['num_words'],
                    'duration': f"{ex['duration']:.3f}",
                    'file_stem': ex['file_stem']
                })
        
        print(f"   - metadata_{name}.csv: {len(examples)} exemples")
    
    def _save_statistics(self, train_examples, test_examples, all_examples):
        """Sauvegarde les statistiques du dataset."""
        stats_path = self.output_dir / 'dataset_stats.txt'
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES DU DATASET TIMIT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Train: {len(train_examples)} exemples\n")
            f.write(f"Test: {len(test_examples)} exemples\n")
            f.write(f"Total: {len(all_examples)} exemples\n\n")
            
            f.write(f"Locuteurs: {len(self.speakers)}\n")
            f.write(f"Transcriptions uniques: {len(self.transcriptions)}\n\n")
            
            # Statistiques par dialecte
            dialects = defaultdict(int)
            for ex in all_examples:
                dialects[ex['dialect_region']] += 1
            
            f.write("Répartition par région dialectale:\n")
            for dialect, count in sorted(dialects.items()):
                f.write(f"  - {dialect}: {count} exemples\n")
            
            # Statistiques par locuteur (top 10)
            speaker_counts = defaultdict(int)
            for ex in all_examples:
                speaker_counts[ex['speaker_id']] += 1
            
            f.write("\nTop 10 locuteurs (par nombre d'exemples):\n")
            for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1])[:10]:
                f.write(f"  - {speaker}: {count} exemples\n")
            
            # Durée moyenne
            durations = [ex['duration'] for ex in all_examples if ex['duration'] > 0]
            if durations:
                f.write(f"\nDurée moyenne: {sum(durations)/len(durations):.2f}s\n")
                f.write(f"Durée min: {min(durations):.2f}s\n")
                f.write(f"Durée max: {max(durations):.2f}s\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   - dataset_stats.txt: sauvegardé")
        
        # Sauvegarder les infos en pickle
        stats_pkl = self.output_dir / 'speakers_info.pkl'
        with open(stats_pkl, 'wb') as f:
            pickle.dump({
                'speakers': list(self.speakers),
                'speaker_to_id': self.speaker_to_id,
                'transcriptions': list(self.transcriptions),
                'transcription_to_id': self.transcription_to_id,
                'total_examples': len(all_examples),
                'train_examples': len(train_examples),
                'test_examples': len(test_examples),
                'num_speakers': len(self.speakers),
                'num_transcriptions': len(self.transcriptions)
            }, f)
        
        print(f"   - speakers_info.pkl: sauvegardé")
    
    def run(self):
        """Exécute tout le pipeline de prétraitement."""
        print("\n" + "="*60)
        print("PRÉTRAITEMENT DU DATASET TIMIT")
        print("="*60)
        self.create_metadata()


def main():
    """Script principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prétraitement du dataset TIMIT")
    parser.add_argument("--raw_dir", type=str, default="./data/raw/timit",
                       help="Dossier contenant les données brutes TIMIT")
    parser.add_argument("--output_dir", type=str, default="./data/processed/timit",
                       help="Dossier de sortie")
    parser.add_argument("--relative_paths", action="store_true",
                       help="Utiliser des chemins relatifs (par défaut: absolus)")
    
    args = parser.parse_args()
    
    preprocessor = TIMITPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        use_absolute_paths=not args.relative_paths
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
