"""
Encodeur acoustique gelé (XLSR/wav2vec2) - utilisé UNIQUEMENT pour
l'extraction initiale des features lors de la construction du graphe.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class FrozenAcousticBackbone(nn.Module):
    """
    Wrapper autour de XLSR/wav2vec2, complètement gelé.
    Utilisé UNIQUEMENT pour extraire les features trame par trame
    lors de la construction du graphe.
    """

    def __init__(self, backbone_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        # Geler tous les paramètres
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            waveforms: (B, T_samples)
            attention_mask: (B, T_samples) - 1 pour les trames valides
        
        Returns:
            frame_features: (B, T_frames, D) -- séquence AVANT pooling
        """
        with torch.no_grad():
            out = self.backbone(waveforms, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, T_frames, D)


def extract_frame_features(
    backbone: FrozenAcousticBackbone,
    waveforms: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    batch_size: int = 16
) -> torch.Tensor:
    """
    Fonction utilitaire pour extraire les features trame par trame
    en batch.
    
    Args:
        backbone: Modèle gelé
        waveforms: (N, T_samples)
        attention_mask: (N, T_samples) ou None
        batch_size: Taille du batch pour l'extraction
    
    Returns:
        frame_features: (N, T_frames, D)
    """
    all_features = []
    n = waveforms.shape[0]
    
    for i in range(0, n, batch_size):
        batch_wav = waveforms[i:i+batch_size]
        batch_mask = attention_mask[i:i+batch_size] if attention_mask is not None else None
        features = backbone(batch_wav, batch_mask)
        all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)
