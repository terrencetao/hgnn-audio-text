"""
merge_multiple_datasets.py
Script pour fusionner plusieurs fichiers CSV en un seul avec colonne source.

Utile pour :
- Combiner plusieurs datasets de langues différentes
- Agréger des données de diverses sources
- Créer un dataset multilingue
"""

import os
import csv
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union
import argparse
import glob


class MultiDatasetMerger:
    """
    Fusionne plusieurs fichiers CSV en un seul.
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        source_names: Optional[List[str]] = None,
        output_dir: str = "./data/processed/merged",
        output_filename: str = "multidataset_merged.csv",
        add_source_column: bool = True,
        shuffle: bool = True,
        random_seed: int = 42,
        deduplicate: bool = False,
        dedup_on: Optional[List[str]] = None,
        auto_align_columns: bool = True,
        normalize_paths: bool = True,
        base_dir: Optional[str] = None,
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        filter_condition: Optional[Dict[str, Union[str, List[str]]]] = None,
        add_language_column: bool = False,
        language_mapping: Optional[Dict[str, str]] = None,
        create_combined_csv: bool = True,
        create_individual_csvs: bool = False
    ):
        """
        Args:
            dataset_paths: Liste des chemins vers les fichiers CSV
            source_names: Noms des sources (si None, utilise les noms de fichiers)
            output_dir: Dossier de sortie
            output_filename: Nom du fichier de sortie
            add_source_column: Ajouter une colonne indiquant la source
            shuffle: Mélanger les données après fusion
            random_seed: Graine aléatoire pour le mélange
            deduplicate: Supprimer les doublons
            dedup_on: Colonnes à utiliser pour la déduplication
            auto_align_columns: Aligner automatiquement les colonnes
            normalize_paths: Normaliser les chemins audio
            base_dir: Dossier de base pour normaliser les chemins
            drop_columns: Colonnes à supprimer
            rename_columns: Dictionnaire de renommage des colonnes
            filter_condition: Conditions de filtrage
            add_language_column: Ajouter une colonne langue
            language_mapping: Mapping source -> langue
            create_combined_csv: Créer le fichier combiné
            create_individual_csvs: Créer des fichiers individuels pour chaque source
        """
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / output_filename
        self.add_source_column = add_source_column
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.deduplicate = deduplicate
        self.dedup_on = dedup_on
        self.auto_align_columns = auto_align_columns
        self.normalize_paths = normalize_paths
        self.base_dir = Path(base_dir) if base_dir else None
        self.drop_columns = drop_columns or []
        self.rename_columns = rename_columns or {}
        self.filter_condition = filter_condition or {}
        self.add_language_column = add_language_column
        self.language_mapping = language_mapping or {}
        self.create_combined_csv = create_combined_csv
        self.create_individual_csvs = create_individual_csvs
        
        # Générer les noms de sources si non fournis
        if source_names is None or len(source_names) != len(dataset_paths):
            self.source_names = []
            for path in dataset_paths:
                # Utiliser le nom du dossier parent ou le nom du fichier
                name = path.parent.name if path.parent.name != '.' else path.stem
                # Nettoyer le nom
                name = name.replace('_metadata', '').replace('_train', '').replace('_test', '')
                self.source_names.append(name)
        else:
            self.source_names = source_names
        
        # Vérifier que les fichiers existent
        for path in self.dataset_paths:
            if not path.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {path}")
        
        print(f"📁 Fusion de {len(dataset_paths)} datasets")
        print(f"   Sources: {self.source_names}")
        print(f"   Sortie: {self.output_path}")
        print(f"   Mélanger: {'Oui' if shuffle else 'Non'}")
        print(f"   Ajouter source: {'Oui' if add_source_column else 'Non'}")
        print(f"   Ajouter langue: {'Oui' if add_language_column else 'Non'}")
    
    def detect_delimiter(self, path: Path) -> str:
        """Détecte le délimiteur d'un fichier CSV."""
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if '\t' in first_line:
                return '\t'
            elif ';' in first_line:
                return ';'
            else:
                return ','
    
    def load_csv(self, path: Path, source_name: str) -> pd.DataFrame:
        """Charge un fichier CSV avec détection automatique."""
        delimiter = self.detect_delimiter(path)
        
        # Tentative avec différents encodages
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Impossible de lire le fichier {path}")
        
        # Normaliser les noms de colonnes
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Ajouter une colonne source
        if self.add_source_column:
            df['source'] = source_name
        
        # Ajouter une colonne langue
        if self.add_language_column:
            if source_name in self.language_mapping:
                df['language'] = self.language_mapping[source_name]
            else:
                df['language'] = source_name  # Utiliser le nom de source comme langue
        
        # Renommer les colonnes
        if self.rename_columns:
            df = df.rename(columns=self.rename_columns)
        
        # Supprimer les colonnes inutiles
        if self.drop_columns:
            existing_cols = [col for col in self.drop_columns if col in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
        
        # Normaliser les chemins
        if self.normalize_paths and 'audio_path' in df.columns:
            df['audio_path'] = df['audio_path'].apply(self.normalize_path)
        
        return df
    
    def normalize_path(self, path: str) -> str:
        """Normalise un chemin audio."""
        if pd.isna(path) or not path:
            return path
        
        path = str(path)
        
        if self.base_dir:
            try:
                abs_path = Path(path)
                if not abs_path.is_absolute():
                    abs_path = self.base_dir / path
                rel_path = abs_path.relative_to(self.base_dir)
                return str(rel_path)
            except ValueError:
                return path
        
        return path
    
    def align_columns(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Aligne les colonnes de tous les dataframes."""
        # Trouver toutes les colonnes
        all_cols = set()
        for df in dataframes:
            all_cols.update(df.columns)
        
        all_cols = sorted(all_cols)
        
        # Ajouter les colonnes manquantes
        aligned_dfs = []
        for df in dataframes:
            for col in all_cols:
                if col not in df.columns:
                    df[col] = None
            aligned_dfs.append(df[all_cols])
        
        return aligned_dfs
    
    def apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique des filtres sur le dataframe."""
        if not self.filter_condition:
            return df
        
        filtered_df = df.copy()
        
        for col, value in self.filter_condition.items():
            if col not in filtered_df.columns:
                print(f"   ⚠️ Colonne '{col}' non trouvée pour le filtrage")
                continue
            
            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[col].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[col] == value]
        
        return filtered_df
    
    def merge(self) -> Dict[str, pd.DataFrame]:
        """
        Fusionne tous les dataframes.
        """
        print("\n📖 Chargement des fichiers...")
        
        dfs = []
        source_stats = {}
        
        for path, source_name in zip(self.dataset_paths, self.source_names):
            print(f"   Chargement: {path} (source: {source_name})")
            df = self.load_csv(path, source_name)
            
            # Appliquer les filtres
            df = self.apply_filter(df)
            
            print(f"      {len(df)} exemples")
            print(f"      Colonnes: {list(df.columns)}")
            
            dfs.append(df)
            source_stats[source_name] = len(df)
        
        # Aligner les colonnes
        if self.auto_align_columns:
            print(f"\n🔧 Alignement des colonnes...")
            dfs = self.align_columns(dfs)
            print(f"   Colonnes finales: {list(dfs[0].columns)}")
        
        # Fusionner
        print(f"\n🔗 Fusion des données...")
        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"   Total avant déduplication: {len(df_combined)} exemples")
        
        # Dédupliquer si demandé
        if self.deduplicate:
            if self.dedup_on:
                dedup_cols = [col for col in self.dedup_on if col in df_combined.columns]
            else:
                if 'audio_path' in df_combined.columns:
                    dedup_cols = ['audio_path']
                elif 'file_stem' in df_combined.columns:
                    dedup_cols = ['file_stem']
                else:
                    dedup_cols = ['transcription'] if 'transcription' in df_combined.columns else None
            
            if dedup_cols:
                print(f"   Déduplication sur: {dedup_cols}")
                initial_len = len(df_combined)
                df_combined = df_combined.drop_duplicates(subset=dedup_cols)
                removed = initial_len - len(df_combined)
                print(f"   Doublons supprimés: {removed}")
                print(f"   Après déduplication: {len(df_combined)} exemples")
        
        # Mélanger si demandé
        if self.shuffle:
            print(f"\n🔀 Mélange des données (seed={self.random_seed})...")
            df_combined = df_combined.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return {
            'combined': df_combined,
            'individual': dfs,
            'stats': source_stats
        }
    
    def save(self, result: Dict[str, Union[pd.DataFrame, List[pd.DataFrame], Dict]]):
        """Sauvegarde les données."""
        df_combined = result['combined']
        dfs = result['individual']
        source_stats = result['stats']
        
        print(f"\n💾 Sauvegarde des fichiers...")
        
        # Sauvegarder le fichier combiné
        if self.create_combined_csv:
            df_combined.to_csv(self.output_path, index=False, encoding='utf-8')
            print(f"   ✅ Fichier combiné: {self.output_path} ({len(df_combined)} exemples)")
        
        # Sauvegarder les fichiers individuels
        if self.create_individual_csvs:
            for df, source_name in zip(dfs, self.source_names):
                individual_path = self.output_dir / f"{source_name}_data.csv"
                df.to_csv(individual_path, index=False, encoding='utf-8')
                print(f"   ✅ Fichier individuel: {individual_path} ({len(df)} exemples)")
        
        # Sauvegarder les statistiques
        stats_path = self.output_dir / "merge_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES DE FUSION MULTI-DATASETS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Nombre de datasets fusionnés: {len(self.dataset_paths)}\n")
            f.write(f"Fichier de sortie: {self.output_path}\n\n")
            
            f.write("Sources fusionnées:\n")
            for source_name, count in source_stats.items():
                f.write(f"  - {source_name}: {count} exemples\n")
            
            f.write(f"\nTotal: {len(df_combined)} exemples\n\n")
            
            # Statistiques par source
            if 'source' in df_combined.columns:
                f.write("Répartition par source:\n")
                source_counts = df_combined['source'].value_counts()
                for source, count in source_counts.items():
                    f.write(f"  - {source}: {count} ({count/len(df_combined):.1%})\n")
            
            # Statistiques par langue
            if 'language' in df_combined.columns:
                f.write("\nRépartition par langue:\n")
                lang_counts = df_combined['language'].value_counts()
                for lang, count in lang_counts.items():
                    f.write(f"  - {lang}: {count} ({count/len(df_combined):.1%})\n")
            
            # Statistiques des locuteurs
            if 'speaker_id' in df_combined.columns:
                f.write(f"\nNombre de locuteurs uniques: {df_combined['speaker_id'].nunique()}\n")
            
            # Statistiques des transcriptions
            if 'transcription' in df_combined.columns:
                non_empty = df_combined['transcription'].notna().sum()
                f.write(f"\nTranscriptions non vides: {non_empty} ({non_empty/len(df_combined):.1%})\n")
                if non_empty > 0:
                    f.write(f"Longueur moyenne: {df_combined['transcription'].str.len().mean():.1f} caractères\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   📊 Statistiques: {stats_path}")
        
        # Afficher le résumé
        print(f"\n📊 Résumé:")
        print(f"   - Total: {len(df_combined)} exemples")
        if 'source' in df_combined.columns:
            source_counts = df_combined['source'].value_counts()
            for source, count in source_counts.items():
                print(f"   - {source}: {count} ({count/len(df_combined):.1%})")


def main():
    """Script principal."""
    parser = argparse.ArgumentParser(
        description="Fusionner plusieurs fichiers CSV"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Liste des chemins vers les fichiers CSV"
    )
    parser.add_argument(
        "--source_names",
        type=str,
        nargs="+",
        help="Noms des sources (optionnel)"
    )
    parser.add_argument(
        "--language_mapping",
        type=str,
        nargs="+",
        help="Mapping source->langue (format: source=langue)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed/merged",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="multidataset_merged.csv",
        help="Nom du fichier de sortie"
    )
    parser.add_argument(
        "--no_source",
        action="store_true",
        help="Ne pas ajouter de colonne source"
    )
    parser.add_argument(
        "--add_language",
        action="store_true",
        help="Ajouter une colonne langue"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Ne pas mélanger les données"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Graine aléatoire"
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Supprimer les doublons"
    )
    parser.add_argument(
        "--dedup_on",
        type=str,
        nargs="+",
        help="Colonnes pour la déduplication"
    )
    parser.add_argument(
        "--drop_columns",
        type=str,
        nargs="+",
        help="Colonnes à supprimer"
    )
    parser.add_argument(
        "--filter_col",
        type=str,
        help="Colonne pour le filtrage"
    )
    parser.add_argument(
        "--filter_value",
        type=str,
        help="Valeur pour le filtrage"
    )
    parser.add_argument(
        "--create_individual",
        action="store_true",
        help="Créer des fichiers individuels par source"
    )
    
    args = parser.parse_args()
    
    # Parser le mapping langue
    language_mapping = {}
    if args.language_mapping:
        for item in args.language_mapping:
            if '=' in item:
                source, lang = item.split('=', 1)
                language_mapping[source] = lang
    
    # Parser les conditions de filtrage
    filter_condition = {}
    if args.filter_col and args.filter_value:
        if ',' in args.filter_value:
            filter_condition[args.filter_col] = [v.strip() for v in args.filter_value.split(',')]
        else:
            filter_condition[args.filter_col] = args.filter_value
    
    merger = MultiDatasetMerger(
        dataset_paths=args.datasets,
        source_names=args.source_names,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        add_source_column=not args.no_source,
        add_language_column=args.add_language,
        language_mapping=language_mapping,
        shuffle=not args.no_shuffle,
        random_seed=args.random_seed,
        deduplicate=args.deduplicate,
        dedup_on=args.dedup_on,
        drop_columns=args.drop_columns,
        filter_condition=filter_condition,
        create_individual_csvs=args.create_individual
    )
    
    result = merger.merge()
    merger.save(result)


if __name__ == "__main__":
    main()
