"""
Normalise les données Yemba (YembaEGRA / YembaTones) vers le format
metadata_csv attendu par data/datasets.py : audio_path, transcription, speaker_id.

Sources (cf. slides "Présentation des jeux de données") :
- YembaEGRA (Kana Azeuko et al., 2024) -- 8031 enregistrements, 60 mots, 69 locuteurs
- YembaTones (Kenfack Jeuguim & Melatagia Yonta, 2023) -- 100 mots pour l'induction
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    # TODO: parser le format brut YembaEGRA (voir doc du dataset), écrire
    # audio_path, transcription, speaker_id dans args.output_csv
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcription", "speaker_id"])
        writer.writeheader()
        # TODO: writer.writerow({...}) pour chaque enregistrement


if __name__ == "__main__":
    main()
