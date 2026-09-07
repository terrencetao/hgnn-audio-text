"""
prepare_commonvoice.py
Script de prétraitement pour le dataset Common Voice.

Structure attendue:
data/raw/zulu/
├── train.tsv
├── test.tsv
├── dev.tsv (optionnel)
└── audio/
    └── common_voice_zu_XXXXX.mp3

Structure de sortie (identique à TIMIT):
data/processed/zulu/
├── audio_wav/           # Fichiers convertis en WAV (optionnel)
│   └── common_voice_zu_XXXXX.wav
├── metadata_train.csv
├── metadata_test.csv
├── metadata_dev.csv (si disponible)
├── metadata_full.csv
├── dataset_stats.txt
└── speakers_info.pkl
"""

import os
import re
import csv
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CommonVoicePreprocessor:
    """
    Prétraitement du dataset Common Voice.
    Format de sortie identique à TIMIT.
    """
    
    def __init__(
        self,
        raw_dir: str = "./data/raw/zulu",
        output_dir: str = "./data/processed/zulu",
        use_absolute_paths: bool = True,
        convert_to_wav: bool = False,
        sample_rate: int = 16000
    ):
        """
        Args:
            raw_dir: Dossier contenant les données brutes Common Voice
            output_dir: Dossier de sortie pour les données traitées
            use_absolute_paths: Si True, utilise des chemins absolus
            convert_to_wav: Si True, convertit les fichiers MP3 en WAV
            sample_rate: Taux d'échantillonnage pour la conversion
        """
        self.raw_dir = Path(raw_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_absolute_paths = use_absolute_paths
        self.convert_to_wav = convert_to_wav
        self.sample_rate = sample_rate
        
        if convert_to_wav:
            self.wav_dir = self.output_dir / 'audio_wav'
            self.wav_dir.mkdir(parents=True, exist_ok=True)
        
        self.examples = []
        self.speakers = set()
        self.transcriptions = set()
        self.speaker_to_id = {}
        self.transcription_to_id = {}
        
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Dossier raw non trouvé: {self.raw_dir}")
        
        print(f"📁 Prétraitement du dataset Common Voice")
        print(f"   Source: {self.raw_dir}")
        print(f"   Destination: {self.output_dir}")
        print(f"   Conversion en WAV: {'Oui' if convert_to_wav else 'Non'}")
        if convert_to_wav:
            print(f"   Taux d'échantillonnage: {sample_rate}Hz")
        print(f"   Chemins absolus: {'Oui' if use_absolute_paths else 'Non'}")
    
    def convert_mp3_to_wav(self, mp3_path: Path, wav_path: Path) -> bool:
        """
        Convertit un fichier MP3 en WAV mono 16kHz.
        Utilise ffmpeg si disponible, sinon librosa.
        """
        try:
            # Essayer avec ffmpeg d'abord (plus rapide)
            cmd = [
                'ffmpeg',
                '-i', str(mp3_path),
                '-ac', '1',           # Mono
                '-ar', str(self.sample_rate),  # Taux d'échantillonnage
                '-y',                 # Écraser si existe
                str(wav_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and wav_path.exists():
                return True
            
            # Si ffmpeg échoue, essayer avec librosa
            try:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(mp3_path, sr=self.sample_rate, mono=True)
                sf.write(wav_path, y, self.sample_rate)
                return True
            except ImportError:
                print(f"   ⚠️ librosa non disponible pour {mp3_path.name}")
                return False
                
        except Exception as e:
            return False
    
    def get_audio_path(self, filename: str) -> Tuple[Path, str, str]:
        """
        Trouve le chemin du fichier audio.
        
        Returns:
            (chemin_audio, nom_fichier, extension)
        """
        filename = filename.strip()
        
        # Déterminer l'extension
        ext = Path(filename).suffix
        if not ext:
            ext = '.mp3'  # Extension par défaut
        
        base_name = Path(filename).stem
        
        # Si conversion en WAV, chercher d'abord le WAV converti
        if self.convert_to_wav:
            wav_path = self.wav_dir / f"{base_name}.wav"
            if wav_path.exists():
                return wav_path, wav_path.name, '.wav'
        
        # Chercher dans les dossiers source
        possible_paths = [
            self.raw_dir / 'audio' / filename,
            self.raw_dir / 'clips' / filename,
            self.raw_dir / filename,
        ]
        
        # Essayer avec différentes extensions
        extensions_to_try = ['.mp3', '.wav', '.m4a', '.flac']
        if ext in extensions_to_try:
            extensions_to_try = [ext] + [e for e in extensions_to_try if e != ext]
        
        for p in possible_paths:
            if p.exists():
                return p, p.name, p.suffix
        
        # Essayer avec différentes extensions
        for ext_try in extensions_to_try:
            for p in possible_paths:
                p_with_ext = p.with_suffix(ext_try)
                if p_with_ext.exists():
                    return p_with_ext, p_with_ext.name, ext_try
        
        return None, filename, ''
    
    def process_audio_file(self, filename: str) -> Tuple[Path, str, bool]:
        """
        Traite un fichier audio : trouve ou convertit en WAV.
        
        Returns:
            (chemin_audio, nom_fichier, conversion_effectuee)
        """
        # Trouver le fichier source
        src_path, src_name, src_ext = self.get_audio_path(filename)
        
        if src_path is None:
            return None, filename, False
        
        # Si conversion en WAV et que ce n'est pas déjà un WAV
        if self.convert_to_wav and src_ext.lower() != '.wav':
            # Chemin de destination
            base_name = src_path.stem
            wav_path = self.wav_dir / f"{base_name}.wav"
            
            # Convertir si le WAV n'existe pas déjà
            if not wav_path.exists():
                success = self.convert_mp3_to_wav(src_path, wav_path)
                if not success:
                    print(f"   ⚠️ Échec de conversion: {src_path.name}")
                    return None, filename, False
            
            return wav_path, wav_path.name, True
        
        return src_path, src_name, False
    
    def parse_tsv(self, tsv_path: Path, split_name: str) -> List[Dict]:
        """
        Parse un fichier TSV Common Voice.
        
        Returns:
            Liste de dictionnaires au format TIMIT
        """
        if not tsv_path.exists():
            print(f"   ⚠️ Fichier TSV non trouvé: {tsv_path}")
            return []
        
        try:
            df = pd.read_csv(tsv_path, sep='\t')
        except Exception as e:
            print(f"   ❌ Erreur lors de la lecture de {tsv_path}: {e}")
            return []
        
        examples = []
        missing_audio = 0
        converted_audio = 0
        
        # Créer une barre de progression
        iterator = tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}")
        
        for idx, row in iterator:
            try:
                client_id = str(row.get('client_id', ''))
                path_str = str(row.get('path', ''))
                sentence = str(row.get('sentence', ''))
                
                # Traiter le fichier audio (trouver ou convertir)
                audio_file, audio_filename, was_converted = self.process_audio_file(path_str)
                
                if audio_file is None:
                    missing_audio += 1
                    if missing_audio <= 5:
                        print(f"\n   ⚠️ Fichier audio non trouvé: {path_str}")
                    continue
                
                if was_converted:
                    converted_audio += 1
                
                # Chemin absolu ou relatif
                if self.use_absolute_paths:
                    audio_path = str(audio_file.resolve())
                else:
                    # Chemin relatif par rapport au dossier raw
                    if self.convert_to_wav and audio_file.parent == self.wav_dir:
                        # Pour les WAV convertis, chemin relatif par rapport au dossier output
                        audio_path = str(audio_file.relative_to(self.output_dir))
                    else:
                        audio_path = str(audio_file.relative_to(self.raw_dir))
                
                # Nettoyer la transcription
                transcription_clean = re.sub(r'[^a-zA-Z0-9\s\'\-\.,!?]', '', sentence)
                
                examples.append({
                    'audio_path': audio_path,
                    'speaker_id': client_id,
                    'split': split_name,
                    'transcription': sentence,
                    'duration': 0.0,
                    'file_stem': Path(audio_filename).stem,
                    'source_file': tsv_path.name,
                    'was_converted': was_converted
                })
                
                if client_id:
                    self.speakers.add(client_id)
                if transcription_clean:
                    self.transcriptions.add(transcription_clean)
                    
            except Exception as e:
                continue
        
        if missing_audio > 0:
            print(f"\n   ⚠️ {missing_audio} fichiers audio manquants sur {len(df)}")
        if converted_audio > 0:
            print(f"   ✅ {converted_audio} fichiers convertis en WAV")
        
        print(f"   ✅ {len(examples)} exemples valides")
        return examples
    
    def load_all_splits(self):
        """Charge tous les fichiers TSV disponibles."""
        print("\n📊 Étape 1: Chargement des splits")
        
        tsv_files = {
            'train': self.raw_dir / 'train.tsv',
            'test': self.raw_dir / 'test.tsv',
            'dev': self.raw_dir / 'dev.tsv',
        }
        
        self.split_examples = {}
        all_examples = []
        
        for split_name, tsv_path in tsv_files.items():
            if tsv_path.exists():
                print(f"\n📄 Traitement de {split_name}.tsv")
                examples = self.parse_tsv(tsv_path, split_name)
                if examples:
                    self.split_examples[split_name] = examples
                    all_examples.extend(examples)
            else:
                print(f"   ⚠️ {split_name}.tsv non trouvé, ignoré")
                self.split_examples[split_name] = []
        
        self.all_examples = all_examples
        
        print(f"\n   Total: {len(all_examples)} exemples")
        
        if not all_examples:
            print("❌ Aucun exemple trouvé!")
            return False
        
        return True
    
    def create_metadata(self):
        """Crée les métadonnées pour chaque split (format TIMIT)."""
        if not hasattr(self, 'all_examples') or not self.all_examples:
            if not self.load_all_splits():
                return
        
        # Créer les mappings
        self.speaker_to_id = {s: i for i, s in enumerate(sorted(self.speakers)) if s}
        self.transcription_to_id = {t: i for i, t in enumerate(sorted(self.transcriptions)) if t}
        
        print(f"\n📝 Étape 2: Création des mappings")
        print(f"   - {len(self.speakers)} locuteurs uniques")
        print(f"   - {len(self.transcriptions)} transcriptions uniques")
        
        # Sauvegarder les métadonnées pour chaque split
        print(f"\n💾 Étape 3: Sauvegarde des métadonnées")
        
        for split_name, examples in self.split_examples.items():
            if examples:
                self._save_metadata(examples, split_name)
        
        # Sauvegarder toutes les données combinées
        if self.all_examples:
            self._save_metadata(self.all_examples, 'full')
        
        # Statistiques
        self._save_statistics()
        
        print(f"\n✅ Prétraitement terminé!")
        print(f"📁 Dossier de sortie: {self.output_dir}")
        if self.convert_to_wav:
            print(f"🎵 Fichiers WAV convertis dans: {self.wav_dir}")
    
    def _save_metadata(self, examples: List[Dict], name: str):
        """Sauvegarde les métadonnées dans un CSV format TIMIT."""
        if not examples:
            print(f"   ⚠️ Pas d'exemples pour {name}")
            return
        
        metadata_path = self.output_dir / f'metadata_{name}.csv'
        
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'audio_path', 'speaker_id', 'split',
                'transcription', 'duration', 'file_stem'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for ex in examples:
                writer.writerow({
                    'audio_path': ex['audio_path'],
                    'speaker_id': ex['speaker_id'],
                    'split': ex['split'],
                    'transcription': ex['transcription'],
                    'duration': f"{ex['duration']:.3f}",
                    'file_stem': ex['file_stem']
                })
        
        print(f"   - metadata_{name}.csv: {len(examples)} exemples")
    
    def _save_statistics(self):
        """Sauvegarde les statistiques du dataset."""
        stats_path = self.output_dir / 'dataset_stats.txt'
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES DU DATASET COMMON VOICE\n")
            f.write("="*60 + "\n\n")
            
            for split_name, examples in self.split_examples.items():
                if examples:
                    f.write(f"{split_name.capitalize()}: {len(examples)} exemples\n")
            
            f.write(f"\nTotal: {len(self.all_examples)} exemples\n\n")
            f.write(f"Locuteurs: {len(self.speakers)}\n")
            f.write(f"Transcriptions uniques: {len(self.transcriptions)}\n")
            
            if self.convert_to_wav:
                converted = sum(1 for ex in self.all_examples if ex.get('was_converted', False))
                f.write(f"Fichiers convertis en WAV: {converted}\n")
            
            # Statistiques par locuteur
            speaker_counts = defaultdict(int)
            for ex in self.all_examples:
                speaker_counts[ex['speaker_id']] += 1
            
            f.write("\nTop 10 locuteurs:\n")
            for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1])[:10]:
                speaker_short = speaker[:16] + "..." if len(speaker) > 20 else speaker
                f.write(f"  - {speaker_short}: {count} exemples\n")
            
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
                'total_examples': len(self.all_examples),
                'num_speakers': len(self.speakers),
                'num_transcriptions': len(self.transcriptions),
                'split_counts': {name: len(ex) for name, ex in self.split_examples.items() if ex}
            }, f)
        
        print(f"   - speakers_info.pkl: sauvegardé")
    
    def run(self):
        """Exécute tout le pipeline de prétraitement."""
        print("\n" + "="*60)
        print("PRÉTRAITEMENT DU DATASET COMMON VOICE")
        print("="*60)
        
        if self.convert_to_wav:
            print("\n🔊 Conversion audio en WAV (cela peut prendre du temps)...")
        
        if not self.load_all_splits():
            return
        
        self.create_metadata()


def main():
    """Script principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prétraitement du dataset Common Voice")
    parser.add_argument("--raw_dir", type=str, default="./data/raw/zulu",
                       help="Dossier contenant les données brutes Common Voice")
    parser.add_argument("--output_dir", type=str, default="./data/processed/zulu",
                       help="Dossier de sortie")
    parser.add_argument("--relative_paths", action="store_true",
                       help="Utiliser des chemins relatifs (par défaut: absolus)")
    parser.add_argument("--convert_to_wav", action="store_true",
                       help="Convertir les fichiers MP3 en WAV mono 16kHz")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Taux d'échantillonnage pour la conversion WAV (défaut: 16000)")
    
    args = parser.parse_args()
    
    preprocessor = CommonVoicePreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        use_absolute_paths=not args.relative_paths,
        convert_to_wav=args.convert_to_wav,
        sample_rate=args.sample_rate
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
