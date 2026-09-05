"""
Dataset PyTorch pour les paires (audio, transcription, locuteur, ...).

Pour la validation croisée, on n'utilise PAS de splits prédéfinis dans le dataset.
Les splits sont gérés par cross_validation.py.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import csv
import os
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


@dataclass
class AudioTextExample:
    """Structure de données pour un exemple audio-texte."""
    waveform: torch.Tensor  # (T_samples,)
    sample_rate: int
    transcription: str
    speaker_id: str
    audio_path: str
    phonemes: Optional[List[str]] = None
    word_timestamps: Optional[List[Tuple[float, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            'waveform': self.waveform,
            'sample_rate': self.sample_rate,
            'transcription': self.transcription,
            'speaker_id': self.speaker_id,
            'audio_path': self.audio_path,
            'phonemes': self.phonemes,
            'word_timestamps': self.word_timestamps,
            'metadata': self.metadata
        }


class AudioTextDataset(Dataset):
    """
    Dataset générique pour des paires audio-texte.
    Pas de splits internes - les splits sont gérés par cross_validation.py.
    
    Args:
        metadata_csv: Chemin vers le fichier CSV avec les colonnes:
                     audio_path, transcription, speaker_id
        sample_rate: Taux d'échantillonnage cible (resample si différent)
        max_length: Longueur maximale en échantillons
        normalize: Normaliser le waveform
        return_path: Retourner le chemin audio en plus
        transform: Transformations supplémentaires à appliquer
    """
    
    def __init__(
        self,
        metadata_csv: str,
        sample_rate: Optional[int] = 16000,
        max_length: Optional[int] = None,
        normalize: bool = True,
        return_path: bool = False,
        transform: Optional[callable] = None
    ):
        self.metadata_csv = metadata_csv
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.normalize = normalize
        self.return_path = return_path
        self.transform = transform
        
        # Charger les métadonnées
        self.examples: List[Dict[str, str]] = []
        self._load_metadata()
        
        # Cache pour les waveforms (optionnel)
        self.cache = {}
        self.use_cache = False
        
        # Vérifier que tous les fichiers existent
        self._validate_files()
    
    def _load_metadata(self):
        """Charge les métadonnées depuis le CSV."""
        with open(self.metadata_csv, newline="", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            required_cols = {"audio_path", "transcription", "speaker_id"}
            if not required_cols.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"CSV doit contenir les colonnes: {required_cols}. "
                    f"Colonnes trouvées: {reader.fieldnames}"
                )
            
            for row in reader:
                self.examples.append(row)
        
        print(f"✅ Dataset chargé: {len(self.examples)} exemples depuis {self.metadata_csv}")
    
    def _validate_files(self):
        """Vérifie que tous les fichiers audio existent."""
        missing = []
        for i, row in enumerate(self.examples):
            if not os.path.exists(row["audio_path"]):
                missing.append((i, row["audio_path"]))
        
        if missing:
            print(f"⚠️ {len(missing)} fichiers audio manquants")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> AudioTextExample:
        """Récupère un exemple."""
        row = self.examples[idx]
        audio_path = row["audio_path"]
        
        # Charger l'audio (avec cache)
        if self.use_cache and audio_path in self.cache:
            waveform, sr = self.cache[audio_path]
        else:
            waveform, sr = torchaudio.load(audio_path)
            if self.use_cache:
                self.cache[audio_path] = (waveform, sr)
        
        # Convertir en mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resampler
        if self.sample_rate is not None and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate
        
        # Squeeze
        waveform = waveform.squeeze(0)
        
        # Tronquer
        if self.max_length is not None and waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        
        # Normaliser
        if self.normalize:
            max_val = waveform.abs().max()
            if max_val > 1e-8:
                waveform = waveform / max_val
        
        # Transformations
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        return AudioTextExample(
            waveform=waveform,
            sample_rate=sr,
            transcription=row["transcription"],
            speaker_id=row["speaker_id"],
            audio_path=audio_path,
            metadata={'original_duration': waveform.shape[0] / sr}
        )
    
    def get_all_transcriptions(self) -> List[str]:
        """Récupère toutes les transcriptions."""
        return [row["transcription"] for row in self.examples]
    
    def get_all_speakers(self) -> List[str]:
        """Récupère tous les IDs de locuteurs."""
        return [row["speaker_id"] for row in self.examples]
    
    def get_unique_transcriptions(self) -> List[str]:
        """Récupère les transcriptions uniques."""
        return list(set(self.get_all_transcriptions()))
    
    def get_unique_speakers(self) -> List[str]:
        """Récupère les IDs de locuteurs uniques."""
        return list(set(self.get_all_speakers()))
    
    def enable_cache(self):
        """Active le cache des waveforms."""
        self.use_cache = True
        print("✅ Cache activé")
    
    def clear_cache(self):
        """Vide le cache."""
        self.cache.clear()
        print("✅ Cache vidé")
    
    def get_indices_by_speaker(self) -> Dict[str, List[int]]:
        """Groupe les indices par locuteur (utile pour CV par locuteur)."""
        groups = {}
        for idx, row in enumerate(self.examples):
            speaker = row["speaker_id"]
            if speaker not in groups:
                groups[speaker] = []
            groups[speaker].append(idx)
        return groups




class Collator:
    """
    Collate function pour DataLoader avec padding à longueur fixe.
    """
    
    def __init__(
        self,
        pad_value: float = 0.0,
        return_attention_mask: bool = True,
        return_indices: bool = True,
        max_length: Optional[int] = None  # Ajout: longueur max fixe
    ):
        self.pad_value = pad_value
        self.return_attention_mask = return_attention_mask
        self.return_indices = return_indices
        self.max_length = max_length
    
    def __call__(self, batch: List[AudioTextExample]) -> Dict[str, Any]:
        """Collate un batch d'exemples."""
        if isinstance(batch[0], tuple):
            batch, indices = zip(*batch)
            indices = list(indices)
        else:
            indices = list(range(len(batch)))
        
        # Trier par longueur pour le padding efficace
        sorted_pairs = sorted(zip(batch, indices), key=lambda x: len(x[0].waveform), reverse=True)
        batch, indices = zip(*sorted_pairs)
        batch = list(batch)
        indices = list(indices)
        
        waveforms = [ex.waveform for ex in batch]
        lengths = [len(w) for w in waveforms]
        
        # Si max_length est spécifié, tronquer et padder à cette longueur
        if self.max_length is not None:
            # Tronquer les waveforms trop longs
            waveforms = [w[:self.max_length] if len(w) > self.max_length else w for w in waveforms]
            lengths = [min(len(w), self.max_length) for w in waveforms]
        
        # Padder
        padded_waveforms = pad_sequence(
            waveforms,
            batch_first=True,
            padding_value=self.pad_value
        )
        
        # Si max_length est spécifié, on peut aussi padder à max_length
        if self.max_length is not None and padded_waveforms.shape[1] < self.max_length:
            # Padder à max_length
            pad_size = self.max_length - padded_waveforms.shape[1]
            padded_waveforms = torch.nn.functional.pad(
                padded_waveforms, 
                (0, pad_size), 
                value=self.pad_value
            )
        
        # Masque d'attention
        if self.return_attention_mask:
            attention_mask = torch.zeros(padded_waveforms.shape, dtype=torch.long)
            for i, length in enumerate(lengths):
                attention_mask[i, :length] = 1
        else:
            attention_mask = None
        
        return {
            'waveforms': padded_waveforms,
            'attention_mask': attention_mask,
            'transcriptions': [ex.transcription for ex in batch],
            'speaker_ids': [ex.speaker_id for ex in batch],
            'audio_paths': [ex.audio_path for ex in batch],
            'lengths': torch.tensor(lengths, dtype=torch.long),
            'indices': torch.tensor(indices, dtype=torch.long),
            'metadata': [ex.metadata for ex in batch],
        }
