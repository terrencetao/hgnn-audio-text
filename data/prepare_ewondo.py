"""
prepare_ewondo.py
Script de prétraitement pour le dataset EWONDO.

Structure attendue:
data/raw/ewondo/
├── transcription.csv          # Fichier avec colonnes: ID, Ewondo
├── loc_001/                   # Dossier du locuteur 1
│   ├── STE-000.wav
│   ├── STE-001.wav
│   └── ...
├── loc_002/
│   └── ...
└── ...

Structure de sortie:
data/processed/ewondo/
├── metadata.csv               # Fichier unique avec toutes les métadonnées
│                              # Les chemins audio sont des chemins ABSOLUS
└── dataset_stats.txt          # Statistiques du dataset
"""

import os
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


class EwondoPreprocessor:
    """
    Prétraitement du dataset EWONDO.
    Pas de splits - tout est dans metadata.csv pour la validation croisée.
    Les chemins audio sont sauvegardés en ABSOLU.
    """
    
    def __init__(
        self,
        raw_dir: str = "./data/raw/ewondo",
        output_dir: str = "./data/processed/ewondo",
        use_absolute_paths: bool = True
    ):
        """
        Args:
            raw_dir: Dossier contenant les données brutes
            output_dir: Dossier de sortie pour les données traitées
            use_absolute_paths: Si True, utilise des chemins absolus dans metadata.csv
        """
        self.raw_dir = Path(raw_dir).resolve()  # Résoudre en chemin absolu
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_absolute_paths = use_absolute_paths
        
        # Initialiser les conteneurs
        self.examples = []
        self.speakers = []
        self.transcriptions = []
        self.transcription_to_id = {}
        self.speaker_to_id = {}
        
        # Vérifier que le dossier raw existe
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Dossier raw non trouvé: {self.raw_dir}")
        
        print(f"📁 Prétraitement du dataset EWONDO")
        print(f"   Source: {self.raw_dir}")
        print(f"   Destination: {self.output_dir}")
        print(f"   Chemins absolus: {'Oui' if use_absolute_paths else 'Non'}")
    
    def load_transcriptions(self) -> Dict[str, str]:
        """
        Charge le fichier transcription.csv.
        
        Format attendu:
        ID,Ewondo
        000,texte en ewondo
        001,autre texte
        ...
        
        Returns:
            Dict {id: transcription}
        """
        csv_path = self.raw_dir / "transcription.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Fichier transcription.csv non trouvé: {csv_path}")
        
        transcriptions = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Détecter le délimiteur
            first_line = f.readline()
            f.seek(0)
            
            if ';' in first_line:
                delimiter = ';'
            else:
                delimiter = ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Identifier les colonnes
            id_col = None
            text_col = None
            
            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if col_lower in ['id', 'index', 'num', 'number', 'ste']:
                    id_col = col
                elif col_lower in ['ewondo', 'text', 'transcription', 'phrase']:
                    text_col = col
            
            if id_col is None or text_col is None:
                raise ValueError(f"Colonnes ID ou transcription non trouvées. "
                               f"Colonnes disponibles: {reader.fieldnames}")
            
            print(f"   Colonne ID: '{id_col}', Colonne transcription: '{text_col}'")
            
            for row in reader:
                # Nettoyer l'ID
                id_str = row[id_col].strip()
                if id_str.startswith('STE-'):
                    id_str = id_str[4:]
                elif id_str.startswith('STE'):
                    id_str = id_str[3:]
                
                # Nettoyer la transcription
                text = row[text_col].strip()
                if text:
                    transcriptions[id_str] = text
        
        print(f"   ✅ {len(transcriptions)} transcriptions chargées")
        return transcriptions
    
    def scan_audio_files(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Parcourt tous les dossiers de locuteurs pour trouver les fichiers audio.
        Les chemins sont retournés en ABSOLU.
        
        Returns:
            Dict {speaker_id: [(absolute_audio_path, audio_id), ...]}
        """
        speaker_files = defaultdict(list)
        
        # Trouver tous les dossiers de locuteurs (loc_XXX)
        loc_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir() and d.name.startswith('loc_')]
        
        if not loc_dirs:
            # Si pas de dossiers loc_, chercher directement les fichiers wav
            wav_files = list(self.raw_dir.glob('*.wav'))
            if wav_files:
                for wav_path in wav_files:
                    # Chemin absolu
                    abs_path = wav_path.resolve()
                    speaker_files['unknown'].append((str(abs_path), wav_path.stem))
                print(f"   Aucun dossier loc_ trouvé, mais {len(wav_files)} fichiers .wav trouvés")
                print(f"   Assignés au locuteur 'unknown'")
            else:
                print(f"   ⚠️ Aucun fichier .wav trouvé dans {self.raw_dir}")
            return speaker_files
        
        for loc_dir in sorted(loc_dirs):
            speaker_id = loc_dir.name
            wav_files = list(loc_dir.glob('*.wav'))
            
            for wav_path in wav_files:
                # Chemin absolu
                abs_path = wav_path.resolve()
                
                # Extraire l'ID du nom de fichier
                filename = wav_path.stem
                if filename.startswith('STE-'):
                    audio_id = filename[4:]
                elif filename.startswith('STE'):
                    audio_id = filename[3:]
                else:
                    audio_id = filename
                
                speaker_files[speaker_id].append((str(abs_path), audio_id))
            
            print(f"   Locuteur {speaker_id}: {len(speaker_files[speaker_id])} fichiers")
        
        return speaker_files
    
    def create_metadata(
        self,
        transcriptions: Dict[str, str],
        speaker_files: Dict[str, List[Tuple[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Crée les métadonnées en faisant correspondre les fichiers audio
        avec leurs transcriptions.
        
        Les chemins audio sont stockés en ABSOLU.
        
        Returns:
            List de dictionnaires avec les colonnes: audio_path, transcription, speaker_id
        """
        metadata = []
        missing_transcriptions = []
        missing_audio = []
        
        for speaker_id, files in speaker_files.items():
            for audio_path, audio_id in files:
                if audio_id in transcriptions:
                    # Vérifier si on utilise des chemins absolus ou relatifs
                    if self.use_absolute_paths:
                        # Déjà absolu car on a utilisé resolve() plus haut
                        final_path = audio_path
                    else:
                        # Convertir en relatif par rapport au dossier de travail
                        final_path = os.path.relpath(audio_path, os.getcwd())
                    
                    metadata.append({
                        'audio_path': final_path,
                        'transcription': transcriptions[audio_id],
                        'speaker_id': speaker_id,
                        'audio_id': audio_id
                    })
                else:
                    missing_transcriptions.append((speaker_id, audio_id))
        
        # Vérifier les transcriptions sans audio
        all_audio_ids = {audio_id for files in speaker_files.values() for _, audio_id in files}
        for audio_id, text in transcriptions.items():
            if audio_id not in all_audio_ids:
                missing_audio.append((audio_id, text))
        
        if missing_transcriptions:
            print(f"   ⚠️ {len(missing_transcriptions)} fichiers sans transcription")
            for speaker, audio_id in missing_transcriptions[:5]:
                print(f"      - {speaker}/{audio_id}")
            if len(missing_transcriptions) > 5:
                print(f"      ... et {len(missing_transcriptions) - 5} autres")
        
        if missing_audio:
            print(f"   ⚠️ {len(missing_audio)} transcriptions sans fichier audio")
            for audio_id, text in missing_audio[:5]:
                print(f"      - {audio_id}: {text[:50]}...")
            if len(missing_audio) > 5:
                print(f"      ... et {len(missing_audio) - 5} autres")
        
        print(f"   ✅ {len(metadata)} paires audio-transcription valides")
        
        # Afficher un exemple de chemin
        if metadata:
            print(f"   Exemple de chemin audio: {metadata[0]['audio_path']}")
        
        return metadata
    
    def save_statistics(self, metadata: List[Dict[str, str]]):
        """
        Sauvegarde les statistiques du dataset.
        """
        # Compter les locuteurs
        speakers = set(exp['speaker_id'] for exp in metadata)
        
        # Compter les transcriptions uniques
        transcriptions = set(exp['transcription'] for exp in metadata)
        
        # Compter par locuteur
        speaker_counts = defaultdict(int)
        for exp in metadata:
            speaker_counts[exp['speaker_id']] += 1
        
        # Compter par transcription
        trans_counts = defaultdict(int)
        for exp in metadata:
            trans_counts[exp['transcription']] += 1
        
        # Compter les fichiers par locuteur et vérifier les chemins
        print("\n   Vérification des chemins audio:")
        for i, exp in enumerate(metadata[:3]):  # Vérifier les 3 premiers
            path = Path(exp['audio_path'])
            exists = path.exists()
            status = "✅" if exists else "❌"
            print(f"      {status} {path}")
        if len(metadata) > 3:
            print(f"      ... et {len(metadata) - 3} autres fichiers")
        
        # Générer le rapport
        stats_path = self.output_dir / 'dataset_stats.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES DU DATASET EWONDO\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Chemin absolu du dataset: {self.output_dir}\n")
            f.write(f"Chemin absolu des fichiers: {self.raw_dir}\n\n")
            
            f.write(f"Nombre total d'exemples: {len(metadata)}\n")
            f.write(f"Nombre de locuteurs: {len(speakers)}\n")
            f.write(f"Nombre de transcriptions uniques: {len(transcriptions)}\n\n")
            
            f.write("Répartition par locuteur:\n")
            for speaker, count in sorted(speaker_counts.items()):
                f.write(f"  - {speaker}: {count} exemples\n")
            
            f.write("\nExemples de chemins absolus:\n")
            for i, exp in enumerate(metadata[:5]):
                f.write(f"  {i+1}. {exp['audio_path']}\n")
            if len(metadata) > 5:
                f.write(f"  ... et {len(metadata) - 5} autres\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   ✅ Statistiques sauvegardées: {stats_path}")
        
        # Sauvegarder les infos en pickle
        stats_pkl_path = self.output_dir / 'speakers_info.pkl'
        with open(stats_pkl_path, 'wb') as f:
            pickle.dump({
                'speakers': list(speakers),
                'speaker_counts': dict(speaker_counts),
                'unique_transcriptions': list(transcriptions),
                'total_examples': len(metadata),
                'num_speakers': len(speakers),
                'raw_dir': str(self.raw_dir),
                'output_dir': str(self.output_dir),
                'use_absolute_paths': self.use_absolute_paths
            }, f)
        print(f"   ✅ Infos locuteurs sauvegardées: {stats_pkl_path}")
    
    def run(self):
        """
        Exécute tout le pipeline de prétraitement.
        """
        print("\n" + "="*60)
        print("PRÉTRAITEMENT DU DATASET EWONDO")
        print("="*60)
        
        # 1. Charger les transcriptions
        print("\n📝 Étape 1: Chargement des transcriptions")
        transcriptions = self.load_transcriptions()
        
        # 2. Scanner les fichiers audio
        print("\n🎵 Étape 2: Scan des fichiers audio")
        speaker_files = self.scan_audio_files()
        
        # 3. Créer les métadonnées
        print("\n📊 Étape 3: Création des métadonnées")
        metadata = self.create_metadata(transcriptions, speaker_files)
        
        if not metadata:
            print("❌ Aucune paire audio-transcription valide trouvée!")
            return
        
        # 4. Sauvegarder le metadata
        print("\n💾 Étape 4: Sauvegarde des métadonnées")
        metadata_path = self.output_dir / 'metadata.csv'
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            if metadata:
                fieldnames = metadata[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata)
        
        print(f"   ✅ Métadonnées sauvegardées: {metadata_path}")
        print(f"      - {len(metadata)} exemples")
        print(f"      - Chemins: {'Absolus' if self.use_absolute_paths else 'Relatifs'}")
        
        # 5. Sauvegarder les statistiques
        print("\n📈 Étape 5: Sauvegarde des statistiques")
        self.save_statistics(metadata)
        
        print("\n" + "="*60)
        print("✅ PRÉTRAITEMENT TERMINÉ!")
        print("="*60)
        print(f"\n📁 Données sauvegardées dans: {self.output_dir}")
        print(f"   - metadata.csv: {len(metadata)} exemples")
        print(f"   - dataset_stats.txt: statistiques détaillées")
        print(f"   - speakers_info.pkl: informations sur les locuteurs")
        print(f"\n📂 Les chemins audio sont en {'ABSOLU' if self.use_absolute_paths else 'RELATIF'}")
        print("\n💡 Pour la validation croisée, les splits seront gérés par cross_validation.py")


def main():
    """Script principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prétraitement du dataset EWONDO")
    parser.add_argument("--raw_dir", type=str, default="./raw/ewondo",
                       help="Dossier contenant les données brutes")
    parser.add_argument("--output_dir", type=str, default="./processed/ewondo",
                       help="Dossier de sortie")
    parser.add_argument("--relative_paths", action="store_true",
                       help="Utiliser des chemins relatifs (par défaut: absolus)")
    
    args = parser.parse_args()
    
    preprocessor = EwondoPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        use_absolute_paths=not args.relative_paths  # Par défaut: absolu
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
