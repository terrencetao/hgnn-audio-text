"""
split_stratified_by_source.py
Script pour effectuer un split train/test stratifié par source avec export par langue.

Produit :
- train.csv / test.csv / validation.csv (fichiers combinés)
- train_LANGUE.csv / test_LANGUE.csv pour chaque langue
- Statistiques détaillées
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple
import argparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import defaultdict


class StratifiedSplitter:
    """
    Effectue un split train/test stratifié par source avec export par langue.
    """
    
    def __init__(
        self,
        input_path: str,
        output_dir: str = "./data/processed/split",
        train_filename: str = "train.csv",
        test_filename: str = "test.csv",
        val_filename: Optional[str] = None,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        val_ratio: float = 0.0,
        split_by: str = "source",
        stratification: bool = True,
        random_seed: int = 42,
        min_examples_per_source: int = 1,
        include_source_column: bool = True,
        include_language_column: bool = True,
        adjust_ratios_by_source: bool = False,
        source_ratios: Optional[Dict[str, Dict[str, float]]] = None,
        create_validation_file: bool = False,
        export_by_language: bool = True,
        language_column: str = "language",
        language_mapping: Optional[Dict[str, str]] = None,
        output_language_prefix: str = "train",
        output_language_suffix: str = "csv",
        combine_by_language: bool = True
    ):
        """
        Args:
            input_path: Chemin vers le fichier CSV à splitter
            output_dir: Dossier de sortie
            train_filename: Nom du fichier train combiné
            test_filename: Nom du fichier test combiné
            val_filename: Nom du fichier validation (optionnel)
            train_ratio: Proportion pour l'entraînement (0-1)
            test_ratio: Proportion pour le test (0-1)
            val_ratio: Proportion pour la validation (0-1)
            split_by: Colonne utilisée pour le split
            stratification: Stratifier par source
            random_seed: Graine aléatoire
            min_examples_per_source: Nombre minimum d'exemples par source
            include_source_column: Inclure la colonne source
            include_language_column: Inclure la colonne langue
            adjust_ratios_by_source: Ajuster les ratios par source
            source_ratios: Ratios personnalisés par source
            create_validation_file: Créer un fichier validation séparé
            export_by_language: Exporter des fichiers séparés par langue
            language_column: Nom de la colonne langue
            language_mapping: Mapping source -> langue
            output_language_prefix: Préfixe pour les fichiers par langue
            output_language_suffix: Suffixe pour les fichiers par langue
            combine_by_language: Combiner les fichiers par langue après split
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins des fichiers combinés
        self.train_path = self.output_dir / train_filename
        self.test_path = self.output_dir / test_filename
        self.val_path = self.output_dir / (val_filename or "validation.csv") if create_validation_file else None
        
        # Paramètres de split
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.split_by = split_by
        self.stratification = stratification
        self.random_seed = random_seed
        self.min_examples_per_source = min_examples_per_source
        self.include_source_column = include_source_column
        self.include_language_column = include_language_column
        self.adjust_ratios_by_source = adjust_ratios_by_source
        self.source_ratios = source_ratios or {}
        self.create_validation_file = create_validation_file
        
        # Paramètres d'export par langue
        self.export_by_language = export_by_language
        self.language_column = language_column
        self.language_mapping = language_mapping or {}
        self.output_language_prefix = output_language_prefix
        self.output_language_suffix = output_language_suffix
        self.combine_by_language = combine_by_language
        
        # Vérifier que le fichier existe
        if not self.input_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {self.input_path}")
        
        # Vérifier que les ratios sont valides
        total_ratio = train_ratio + test_ratio + val_ratio
        if not np.isclose(total_ratio, 1.0):
            print(f"⚠️ La somme des ratios ({total_ratio}) n'est pas égale à 1.0")
            print(f"   Normalisation en cours...")
            self.train_ratio = train_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
            self.val_ratio = val_ratio / total_ratio
        
        print(f"📁 Split stratifié par source")
        print(f"   Fichier d'entrée: {self.input_path}")
        print(f"   Split par: {split_by}")
        print(f"   Ratios: train={self.train_ratio:.1%}, test={self.test_ratio:.1%}")
        if self.val_ratio > 0:
            print(f"           val={self.val_ratio:.1%}")
        print(f"   Stratification: {'Oui' if stratification else 'Non'}")
        print(f"   Export par langue: {'Oui' if export_by_language else 'Non'}")
    
    def detect_delimiter(self, path: Path) -> str:
        """Détecte le délimiteur."""
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if '\t' in first_line:
                return '\t'
            elif ';' in first_line:
                return ';'
            else:
                return ','
    
    def load_data(self) -> pd.DataFrame:
        """Charge les données."""
        delimiter = self.detect_delimiter(self.input_path)
        
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(self.input_path, delimiter=delimiter, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Impossible de lire le fichier {self.input_path}")
        
        # Normaliser les noms de colonnes
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # S'assurer que la colonne langue existe
        if self.export_by_language and self.language_column not in df.columns:
            print(f"⚠️ Colonne '{self.language_column}' non trouvée")
            # Essayer de créer la colonne langue à partir du mapping
            if 'source' in df.columns and self.language_mapping:
                df[self.language_column] = df['source'].map(self.language_mapping)
                # Remplir les valeurs manquantes
                df[self.language_column] = df[self.language_column].fillna(df['source'])
                print(f"   Colonne '{self.language_column}' créée à partir du mapping source->langue")
            else:
                # Utiliser 'source' comme langue
                if 'source' in df.columns:
                    df[self.language_column] = df['source']
                    print(f"   Utilisation de 'source' comme colonne langue")
                else:
                    # Créer une colonne langue par défaut
                    df[self.language_column] = 'unknown'
                    print(f"   Colonne '{self.language_column}' créée avec 'unknown'")
        
        print(f"   {len(df)} exemples chargés")
        print(f"   Colonnes: {list(df.columns)}")
        
        if self.export_by_language and self.language_column in df.columns:
            languages = df[self.language_column].unique()
            print(f"   Langues trouvées: {list(languages)}")
            for lang in languages:
                count = len(df[df[self.language_column] == lang])
                print(f"      - {lang}: {count} exemples")
        
        return df
    
    def get_stratification_column(self, df: pd.DataFrame) -> str:
        """Détermine la colonne à utiliser pour la stratification."""
        if self.split_by in df.columns:
            return self.split_by
        elif self.split_by == "source" and "source" in df.columns:
            return "source"
        elif self.split_by == "speaker" and "speaker_id" in df.columns:
            return "speaker_id"
        elif self.split_by == "language" and self.language_column in df.columns:
            return self.language_column
        else:
            # Utiliser la colonne source par défaut
            if "source" in df.columns:
                return "source"
            elif "speaker_id" in df.columns:
                return "speaker_id"
            elif self.language_column in df.columns:
                return self.language_column
            else:
                print(f"⚠️ Colonne '{self.split_by}' non trouvée")
                print(f"   Utilisation de la première colonne disponible")
                return df.columns[0]
    
    def split_data(self, df: pd.DataFrame, stratify_col: str) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Effectue le split des données.
        """
        # Si la colonne de stratification n'existe pas, la créer
        if stratify_col not in df.columns:
            if self.include_source_column and "source" in df.columns:
                stratify_col = "source"
            elif self.language_column in df.columns:
                stratify_col = self.language_column
            elif "speaker_id" in df.columns:
                stratify_col = "speaker_id"
            else:
                stratify_col = None
        
        # Grouper par la colonne de stratification
        if stratify_col:
            grouped = df.groupby(stratify_col)
        else:
            grouped = df.groupby(lambda x: 0)
        
        train_data = []
        test_data = []
        val_data = []
        
        source_splits = {}
        
        for group_name, group_df in grouped:
            # Vérifier le nombre minimum d'exemples
            if len(group_df) < self.min_examples_per_source:
                print(f"   ⚠️ Groupe '{group_name}' a seulement {len(group_df)} exemples")
                print(f"      Assigné entièrement à l'entraînement")
                train_data.append(group_df)
                source_splits[group_name] = {'train': len(group_df), 'test': 0, 'val': 0}
                continue
            
            # Vérifier si des ratios personnalisés sont définis
            if self.adjust_ratios_by_source and group_name in self.source_ratios:
                ratios = self.source_ratios[group_name]
                train_r = ratios.get('train', self.train_ratio)
                test_r = ratios.get('test', self.test_ratio)
                val_r = ratios.get('val', self.val_ratio)
            else:
                train_r = self.train_ratio
                test_r = self.test_ratio
                val_r = self.val_ratio
            
            # Split
            indices = np.arange(len(group_df))
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            
            train_idx = indices[:int(len(indices) * train_r)]
            remaining = indices[int(len(indices) * train_r):]
            
            if val_r > 0 and len(remaining) > 0:
                val_size = val_r / (val_r + test_r)
                val_idx = remaining[:int(len(remaining) * val_size)]
                test_idx = remaining[int(len(remaining) * val_size):]
            else:
                val_idx = []
                test_idx = remaining
            
            # Récupérer les données
            train_data.append(group_df.iloc[train_idx])
            test_data.append(group_df.iloc[test_idx])
            if len(val_idx) > 0:
                val_data.append(group_df.iloc[val_idx])
            
            source_splits[group_name] = {
                'train': len(train_idx),
                'test': len(test_idx),
                'val': len(val_idx)
            }
        
        # Combiner tous les groupes
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        
        return {
            'train': train_df,
            'test': test_df,
            'val': val_df,
            'source_splits': source_splits
        }
    
    def split(self) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Exécute le split."""
        df = self.load_data()
        
        # Déterminer la colonne de stratification
        stratify_col = self.get_stratification_column(df)
        print(f"\n🔀 Split des données par: {stratify_col}")
        
        # Effectuer le split
        result = self.split_data(df, stratify_col)
        
        return result
    
    def _export_by_language(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Exporte des fichiers séparés par langue.
        """
        if not self.export_by_language:
            return {}
        
        # Vérifier que la colonne langue existe
        if self.language_column not in train_df.columns:
            print(f"⚠️ Colonne '{self.language_column}' non trouvée pour l'export par langue")
            return {}
        
        # Créer un dossier pour les fichiers par langue
        lang_dir = self.output_dir / "by_language"
        lang_dir.mkdir(exist_ok=True)
        
        print(f"\n🌍 Export par langue dans: {lang_dir}")
        
        # Trouver toutes les langues
        all_languages = set()
        for df in [train_df, test_df, val_df]:
            if len(df) > 0 and self.language_column in df.columns:
                all_languages.update(df[self.language_column].unique())
        
        stats_by_language = {}
        
        for language in sorted(all_languages):
            # Filtrer les données pour cette langue
            train_lang = train_df[train_df[self.language_column] == language] if len(train_df) > 0 else pd.DataFrame()
            test_lang = test_df[test_df[self.language_column] == language] if len(test_df) > 0 else pd.DataFrame()
            val_lang = val_df[val_df[self.language_column] == language] if len(val_df) > 0 else pd.DataFrame()
            
            # Statistiques
            stats_by_language[language] = {
                'train': len(train_lang),
                'test': len(test_lang),
                'val': len(val_lang)
            }
            
            # Exporter les fichiers
            if len(train_lang) > 0:
                train_path = lang_dir / f"train_{language}.{self.output_language_suffix}"
                train_lang.to_csv(train_path, index=False, encoding='utf-8')
                print(f"   ✅ Train ({language}): {train_path} ({len(train_lang)} exemples)")
            
            if len(test_lang) > 0:
                test_path = lang_dir / f"test_{language}.{self.output_language_suffix}"
                test_lang.to_csv(test_path, index=False, encoding='utf-8')
                print(f"   ✅ Test ({language}): {test_path} ({len(test_lang)} exemples)")
            
            if len(val_lang) > 0:
                val_path = lang_dir / f"val_{language}.{self.output_language_suffix}"
                val_lang.to_csv(val_path, index=False, encoding='utf-8')
                print(f"   ✅ Val ({language}): {val_path} ({len(val_lang)} exemples)")
        
        # Exporter aussi une version combinée par langue (fusionner train+test+val)
        if self.combine_by_language:
            for language in sorted(all_languages):
                # Combiner toutes les données de cette langue
                lang_dfs = []
                for df in [train_df, test_df, val_df]:
                    if len(df) > 0:
                        lang_part = df[df[self.language_column] == language]
                        if len(lang_part) > 0:
                            lang_dfs.append(lang_part)
                
                if lang_dfs:
                    lang_combined = pd.concat(lang_dfs, ignore_index=True)
                    combined_path = lang_dir / f"{language}_all.{self.output_language_suffix}"
                    lang_combined.to_csv(combined_path, index=False, encoding='utf-8')
                    print(f"   ✅ Combined ({language}): {combined_path} ({len(lang_combined)} exemples)")
        
        # Sauvegarder les statistiques par langue
        stats_path = lang_dir / "language_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES PAR LANGUE\n")
            f.write("="*60 + "\n\n")
            
            for language, counts in sorted(stats_by_language.items()):
                f.write(f"\n{language}:\n")
                f.write(f"  - Train: {counts['train']}\n")
                f.write(f"  - Test: {counts['test']}\n")
                if counts['val'] > 0:
                    f.write(f"  - Validation: {counts['val']}\n")
                f.write(f"  - Total: {sum(counts.values())}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   📊 Statistiques par langue: {stats_path}")
        
        return stats_by_language
    
    def save(self, result: Dict[str, Union[pd.DataFrame, Dict]]):
        """Sauvegarde les fichiers splités."""
        train_df = result['train']
        test_df = result['test']
        val_df = result.get('val', pd.DataFrame())
        source_splits = result.get('source_splits', {})
        
        print(f"\n💾 Sauvegarde des fichiers...")
        
        # Sauvegarder les fichiers combinés
        train_df.to_csv(self.train_path, index=False, encoding='utf-8')
        print(f"   ✅ Train (combiné): {self.train_path} ({len(train_df)} exemples)")
        
        test_df.to_csv(self.test_path, index=False, encoding='utf-8')
        print(f"   ✅ Test (combiné): {self.test_path} ({len(test_df)} exemples)")
        
        if self.create_validation_file and len(val_df) > 0:
            val_df.to_csv(self.val_path, index=False, encoding='utf-8')
            print(f"   ✅ Validation (combiné): {self.val_path} ({len(val_df)} exemples)")
        
        # Exporter par langue
        if self.export_by_language:
            lang_stats = self._export_by_language(train_df, test_df, val_df)
        else:
            lang_stats = {}
        
        # Sauvegarder les statistiques globales
        stats_path = self.output_dir / "split_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STATISTIQUES DE SPLIT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Fichier d'entrée: {self.input_path}\n")
            f.write(f"Split par: {self.split_by}\n")
            f.write(f"Ratios: train={self.train_ratio:.1%}, test={self.test_ratio:.1%}")
            if self.val_ratio > 0:
                f.write(f", val={self.val_ratio:.1%}")
            f.write(f"\n\n")
            
            total = len(train_df) + len(test_df) + len(val_df)
            f.write(f"Nombre total: {total} exemples\n")
            f.write(f"  - Train: {len(train_df)} ({len(train_df)/total:.1%})\n")
            f.write(f"  - Test: {len(test_df)} ({len(test_df)/total:.1%})\n")
            if len(val_df) > 0:
                f.write(f"  - Validation: {len(val_df)} ({len(val_df)/total:.1%})\n")
            
            # Statistiques par source
            if source_splits:
                f.write(f"\nRépartition par {self.split_by}:\n")
                for source, counts in source_splits.items():
                    f.write(f"  - {source}: {counts['train']} train")
                    if counts.get('val', 0) > 0:
                        f.write(f", {counts['val']} val")
                    f.write(f", {counts['test']} test\n")
            
            # Statistiques par langue (fichiers combinés)
            if self.language_column in train_df.columns:
                f.write(f"\nRépartition des langues (combiné):\n")
                all_languages = set()
                for df in [train_df, test_df, val_df]:
                    if len(df) > 0:
                        all_languages.update(df[self.language_column].unique())
                
                for lang in sorted(all_languages):
                    train_count = len(train_df[train_df[self.language_column] == lang]) if len(train_df) > 0 else 0
                    test_count = len(test_df[test_df[self.language_column] == lang]) if len(test_df) > 0 else 0
                    val_count = len(val_df[val_df[self.language_column] == lang]) if len(val_df) > 0 else 0
                    total_count = train_count + test_count + val_count
                    f.write(f"  - {lang}: {total_count} total")
                    if train_count > 0:
                        f.write(f" (train: {train_count}")
                    if test_count > 0:
                        f.write(f", test: {test_count}")
                    if val_count > 0:
                        f.write(f", val: {val_count}")
                    if train_count > 0 or test_count > 0 or val_count > 0:
                        f.write(")")
                    f.write("\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   📊 Statistiques globales: {stats_path}")
        
        # Afficher le résumé
        print(f"\n📊 Résumé du split:")
        print(f"   - Train (combiné): {len(train_df)} exemples")
        print(f"   - Test (combiné): {len(test_df)} exemples")
        if len(val_df) > 0:
            print(f"   - Validation (combiné): {len(val_df)} exemples")
        
        if self.export_by_language and self.language_column in train_df.columns:
            print(f"\n   Langues présentes:")
            all_languages = set()
            for df in [train_df, test_df, val_df]:
                if len(df) > 0:
                    all_languages.update(df[self.language_column].unique())
            
            for lang in sorted(all_languages):
                train_count = len(train_df[train_df[self.language_column] == lang]) if len(train_df) > 0 else 0
                test_count = len(test_df[test_df[self.language_column] == lang]) if len(test_df) > 0 else 0
                val_count = len(val_df[val_df[self.language_column] == lang]) if len(val_df) > 0 else 0
                print(f"      - {lang}: train={train_count}, test={test_count}, val={val_count}")


def main():
    """Script principal."""
    parser = argparse.ArgumentParser(
        description="Split stratifié par source avec export par langue"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Fichier CSV d'entrée"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed/split",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion pour l'entraînement"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Proportion pour le test"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Proportion pour la validation"
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default="source",
        help="Colonne utilisée pour le split (source, speaker, language, ...)"
    )
    parser.add_argument(
        "--language_column",
        type=str,
        default="language",
        help="Nom de la colonne langue"
    )
    parser.add_argument(
        "--no_stratification",
        action="store_true",
        help="Désactiver la stratification"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Graine aléatoire"
    )
    parser.add_argument(
        "--min_examples",
        type=int,
        default=1,
        help="Nombre minimum d'exemples par source"
    )
    parser.add_argument(
        "--create_validation",
        action="store_true",
        help="Créer un fichier validation séparé"
    )
    parser.add_argument(
        "--source_ratios",
        type=str,
        nargs="+",
        help="Ratios personnalisés par source (format: source=train,test,val)"
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default="train.csv",
        help="Nom du fichier train"
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        default="test.csv",
        help="Nom du fichier test"
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default="validation.csv",
        help="Nom du fichier validation"
    )
    parser.add_argument(
        "--no_export_by_language",
        action="store_true",
        help="Ne pas exporter par langue"
    )
    parser.add_argument(
        "--language_prefix",
        type=str,
        default="train",
        help="Préfixe pour les fichiers par langue"
    )
    parser.add_argument(
        "--language_suffix",
        type=str,
        default="csv",
        help="Suffixe pour les fichiers par langue"
    )
    parser.add_argument(
        "--no_combine_language",
        action="store_true",
        help="Ne pas créer de fichiers combinés par langue"
    )
    
    args = parser.parse_args()
    
    # Parser les ratios par source
    source_ratios = {}
    if args.source_ratios:
        for item in args.source_ratios:
            if '=' in item:
                source, ratios = item.split('=', 1)
                parts = ratios.split(',')
                if len(parts) == 2:
                    source_ratios[source] = {
                        'train': float(parts[0]),
                        'test': float(parts[1])
                    }
                elif len(parts) == 3:
                    source_ratios[source] = {
                        'train': float(parts[0]),
                        'test': float(parts[1]),
                        'val': float(parts[2])
                    }
    
    splitter = StratifiedSplitter(
        input_path=args.input,
        output_dir=args.output_dir,
        train_filename=args.train_filename,
        test_filename=args.test_filename,
        val_filename=args.val_filename,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        split_by=args.split_by,
        stratification=not args.no_stratification,
        random_seed=args.random_seed,
        min_examples_per_source=args.min_examples,
        include_source_column=True,
        include_language_column=True,
        adjust_ratios_by_source=bool(source_ratios),
        source_ratios=source_ratios if source_ratios else None,
        create_validation_file=args.create_validation,
        export_by_language=not args.no_export_by_language,
        language_column=args.language_column,
        output_language_prefix=args.language_prefix,
        output_language_suffix=args.language_suffix,
        combine_by_language=not args.no_combine_language
    )
    
    result = splitter.split()
    splitter.save(result)


if __name__ == "__main__":
    main()
