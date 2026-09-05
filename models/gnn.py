"""
Encodeur GraphSAGE hétérogène avec AttentivePooling intégré.
Détection automatique des features et adaptation dynamique des dimensions.

Règles d'adaptation:
1. Audio en trames + Word poolé → hidden = dimension de Word (poolé)
2. Audio poolé + Word en trames → hidden = dimension de Audio (poolé)
3. Les deux en trames → hidden = 768 (standard)
4. Les deux poolés → hidden = 768 (standard)
5. Un seul présent → hidden = 768 (standard)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv


class AttentivePooling(nn.Module):
    """
    Pondération apprise des trames temporelles.
    Détection automatique : si les features sont déjà poolées, retourne telles quelles.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_proj = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) - Features trames OU (B, D) - Features poolées
            attention_mask: (B, T) ou None - Ignoré si features déjà poolées
        
        Returns:
            pooled: (B, D) - Features poolées
        """
        # Détection automatique : si features a 2 dimensions, c'est déjà poolé
        if features.dim() == 2:
            # Déjà poolé, on retourne tel quel
            return features
        
        # Sinon, c'est des trames (B, T, D)
        if features.dim() != 3:
            raise ValueError(f"Features doit être (B, D) ou (B, T, D), mais a {features.shape}")
        
        # Calcul des scores d'attention
        scores = self.attention_proj(features).squeeze(-1)  # (B, T)
        
        # Appliquer le masque d'attention si fourni
        if attention_mask is not None:
            if attention_mask.shape[1] != scores.shape[1]:
                if attention_mask.shape[1] > scores.shape[1]:
                    attention_mask = attention_mask[:, :scores.shape[1]]
                else:
                    pad_size = scores.shape[1] - attention_mask.shape[1]
                    pad = torch.ones(attention_mask.shape[0], pad_size, 
                                    dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, pad], dim=1)
            
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        
        # Softmax et pooling pondéré
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.einsum("bt,btd->bd", weights, features)  # (B, D)
        
        return pooled


class HeterogeneousGNN(nn.Module):
    """
    GNN hétérogène avec AttentivePooling intégré.
    Adaptation dynamique des dimensions selon les features reçues.
    """
    
    def __init__(
        self,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Dimensions déterminées dynamiquement pendant le forward
        self.hidden_dim = None
        self.audio_feature_dim = None
        self.linguistic_feature_dim = None
        
        # Projections créées dynamiquement
        self.audio_proj = None
        self.linguistic_proj = None
        self.input_proj = None
        
        # AttentivePooling (sera initialisé dynamiquement)
        #self.pooling = None
        self.pooling_ling = None
        self.pooling_audio=None
        
        # Couches de message passing (seront initialisées dynamiquement)
        self.convs = nn.ModuleList()
        self.n_layers = n_layers
        self.dropout_rate = dropout
        
        self.activation = nn.ReLU()
        self.dropout = None  # Sera initialisé dynamiquement
    
    def _initialize_dynamic_layers(self, hidden_dim: int, audio_dim: int, linguistic_dim: int):
        """
        Initialise les couches dynamiquement selon les dimensions détectées.
        """
        self.hidden_dim = hidden_dim
        self.audio_feature_dim = audio_dim
        self.linguistic_feature_dim = linguistic_dim
        
        # Projections pour aligner les dimensions
        if audio_dim != hidden_dim and audio_dim > 0:
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        else:
            self.audio_proj = nn.Identity()
        
        if linguistic_dim != hidden_dim and linguistic_dim > 0:
            self.linguistic_proj = nn.Linear(linguistic_dim, hidden_dim)
        else:
            self.linguistic_proj = nn.Identity()
        
        # AttentivePooling avec la bonne dimension
        self.pooling_audio = AttentivePooling(audio_dim)
        self.pooling_ling = AttentivePooling(linguistic_dim)
        
        # Projection initiale
        #self.input_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Couches de message passing
        self.convs = nn.ModuleList()
        for _ in range(self.n_layers):
            conv = HeteroConv(
                {
                    ("audio", "similar_to", "audio"): SAGEConv(hidden_dim, hidden_dim),
                    #("word", "similar_to", "word"): SAGEConv(hidden_dim, hidden_dim),
                    ("audio", "transcribed_as", "word"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                    ("word", "rev_transcribed_as", "audio"): SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    ),
                },
                aggr="mean",
            )
            self.convs.append(conv)
        
        # Déplacer les nouvelles couches sur le bon device
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        self.to(device)
        
        print(f"   ✅ GNN initialisé avec hidden_dim={hidden_dim}")
        print(f"      - Audio: {audio_dim} → {hidden_dim}")
        print(f"      - Word: {linguistic_dim} → {hidden_dim}")
    
    def _detect_dimensions(self, features_dict: dict) -> tuple[int, int, int]:
        """
        Détecte les dimensions des features et détermine la dimension cachée.
        
        Règles:
        1. Audio en trames + Word poolé → hidden = dimension de Word (poolé)
        2. Audio poolé + Word en trames → hidden = dimension de Audio (poolé)
        3. Les deux en trames → hidden = 768 (dimension standard)
        4. Les deux poolés → hidden = 768 (dimension standard)
        5. Un seul présent → hidden = 768 (dimension standard)
        
        Returns:
            hidden_dim: Dimension cachée
            audio_dim: Dimension des features audio
            linguistic_dim: Dimension des features linguistiques
        """
        audio_dim = None
        linguistic_dim = None
        audio_is_pooled = False
        word_is_pooled = False
        
        # Détecter la dimension audio
        if 'audio' in features_dict and features_dict['audio'] is not None:
            features = features_dict['audio']
            if features.dim() == 2:
                audio_dim = features.shape[-1]
                audio_is_pooled = True
            elif features.dim() == 3:
                audio_dim = features.shape[-1]
                audio_is_pooled = False
            else:
                raise ValueError(f"Dimension audio non supportée: {features.shape}")
            print(f"   Audio: dim={audio_dim}, {'poolé' if audio_is_pooled else 'trames'}")
        
        # Détecter la dimension linguistique
        if 'word' in features_dict and features_dict['word'] is not None:
            features = features_dict['word']
            if features.dim() == 2:
                linguistic_dim = features.shape[-1]
                word_is_pooled = True
            elif features.dim() == 3:
                linguistic_dim = features.shape[-1]
                word_is_pooled = False
            else:
                raise ValueError(f"Dimension word non supportée: {features.shape}")
            print(f"   Word: dim={linguistic_dim}, {'poolé' if word_is_pooled else 'trames'}")
        
        # Déterminer la dimension cachée selon les règles
        if audio_dim is not None and linguistic_dim is not None:
            # Les deux sont présents
            if not audio_is_pooled and word_is_pooled:
                # Cas 1: Audio en trames, Word poolé → hidden = dimension de Word
                hidden_dim = linguistic_dim
                print(f"   → Audio en trames ({audio_dim}), Word poolé ({linguistic_dim})")
                print(f"   → hidden={hidden_dim} (aligné sur Word poolé)")
            
            elif audio_is_pooled and not word_is_pooled:
                # Cas 2: Audio poolé, Word en trames → hidden = dimension de Audio
                hidden_dim = audio_dim
                print(f"   → Audio poolé ({audio_dim}), Word en trames ({linguistic_dim})")
                print(f"   → hidden={hidden_dim} (aligné sur Audio poolé)")
            
            else:
                # Cas 3: Les deux en trames → 768
                # Cas 4: Les deux poolés → 768
                hidden_dim = 768
                if not audio_is_pooled and not word_is_pooled:
                    print(f"   → Les deux en trames: audio={audio_dim}, word={linguistic_dim}")
                    print(f"   → hidden=768 (standard)")
                else:
                    print(f"   → Les deux poolés: audio={audio_dim}, word={linguistic_dim}")
                    print(f"   → hidden=768 (standard)")
        
        elif audio_dim is not None:
            # Seulement audio présent → 768
            hidden_dim = 768
            print(f"   → Seulement audio présent: dimension={audio_dim}")
            print(f"   → hidden=768 (standard)")
        
        elif linguistic_dim is not None:
            # Seulement word présent → 768
            hidden_dim = 768
            print(f"   → Seulement word présent: dimension={linguistic_dim}")
            print(f"   → hidden=768 (standard)")
        
        else:
            # Aucune feature → 768
            hidden_dim = 768
            print(f"   → Aucune feature détectée")
            print(f"   → hidden=768 (par défaut)")
        
        return hidden_dim, audio_dim or 0, linguistic_dim or 0
    
    def _get_num_nodes(self, node_type: str, edge_index_dict: dict) -> int:
        """
        Récupère le nombre de nœuds d'un type donné à partir des edge_index.
        """
        if node_type == "audio":
            edge_index = edge_index_dict.get(('audio', 'similar_to', 'audio'), None)
        elif node_type == "word":
            edge_index = edge_index_dict.get(('word', 'similar_to', 'word'), None)
        else:
            return 0
        
        if edge_index is not None and edge_index.numel() > 0:
            return edge_index.max().item() + 1
        return 0
    
    def _process_node_features(
        self,
        node_type: str,
        features: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        edge_index_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Traite les features d'un type de nœud.
        Détection automatique : pooling si nécessaire.
        Projection à la dimension hidden_dim.
        """
        device = next(self.parameters()).device
        
        if features is None:
            # Pas de features pour ce type → zéros
            n_nodes = self._get_num_nodes(node_type, edge_index_dict)
            if n_nodes == 0:
                return None, None
            pooled = torch.zeros(n_nodes, self.hidden_dim, device=device)
        else:
            # Appliquer la projection appropriée selon le type
            #if node_type == "audio" and self.audio_proj is not None:
            #    features = self.audio_proj(features)
            #elif node_type == "word" and self.linguistic_proj is not None:
            #    features = self.linguistic_proj(features)
            
            # Le pooling détecte automatiquement si c'est déjà poolé ou non
            if node_type == "audio":
                pooled = self.pooling_audio(features, attention_mask)
                projected = self.audio_proj(pooled)
            elif node_type == "word":
                pooled = self.pooling_ling(features, attention_mask)
                projected = self.linguistic_proj(pooled)
        
        # Projection finale pour homogénéiser
        #print(f'pooled shape :{pooled.shape}')
        
        
        return pooled, projected
    
    def forward(
        self,
        features_dict: dict[str, torch.Tensor],
        attention_mask_dict: dict[str, torch.Tensor | None] | None = None,
        edge_index_dict: dict | None = None,
        return_pooled: bool = False
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Args:
            features_dict: {
                "audio": (B_audio, T_frames, D_audio) OU (B_audio, D_audio),
                "word": (B_word, D_ling) OU (B_word, T_frames, D_ling)
            }
            attention_mask_dict: {
                "audio": (B_audio, T_frames) ou None,
                "word": (B_word, T_frames) ou None
            }
            edge_index_dict: clé = (src_type, rel, dst_type), valeur = edge_index (2, E)
            return_pooled: Si True, retourne aussi les vecteurs poolés
        
        Returns:
            x_dict: {
                "audio": (N_audio, hidden_dim) - après message passing,
                "word": (N_word, hidden_dim) - après message passing
            }
            pooled_dict (optionnel): {
                "audio": (N_audio, hidden_dim) - avant message passing,
                "word": (N_word, hidden_dim) - avant message passing
            }
        """
        if attention_mask_dict is None:
            attention_mask_dict = {}
        
        if edge_index_dict is None:
            raise ValueError("edge_index_dict est requis")
        
        print("\n🔍 Détection des dimensions des features:")
        
        # Détecter les dimensions et initialiser les couches si nécessaire
        hidden_dim, audio_dim, linguistic_dim = self._detect_dimensions(features_dict)
        
        # Si les couches ne sont pas initialisées ou si les dimensions ont changé
        if (self.hidden_dim != hidden_dim or 
            self.audio_feature_dim != audio_dim or 
            self.linguistic_feature_dim != linguistic_dim):
            self._initialize_dynamic_layers(hidden_dim, audio_dim, linguistic_dim)
        
        x_dict = {}
        pooled_dict = {}
        
        # Traiter chaque type de nœud avec détection automatique
        for node_type in ["audio", "word"]:
            features = features_dict.get(node_type, None)
            attention_mask = attention_mask_dict.get(node_type, None)
            
            pooled, projected = self._process_node_features(
                node_type, features, attention_mask, edge_index_dict
            )
            
            if pooled is not None and projected is not None:
                pooled_dict[node_type] = pooled
                x_dict[node_type] = projected
        
        # Si un type de nœud n'a pas été ajouté, le faire avec des zéros
        for node_type in ["audio", "word"]:
            if node_type not in x_dict:
                n_nodes = self._get_num_nodes(node_type, edge_index_dict)
                if n_nodes > 0:
                    device = next(self.parameters()).device
                    pooled = torch.zeros(n_nodes, self.hidden_dim, device=device)
                    pooled_dict[node_type] = pooled
                    x_dict[node_type] = self.input_proj(pooled)
        
        # Vérification que les deux types sont présents
        if "audio" not in x_dict:
            raise ValueError("Aucun nœud 'audio' trouvé dans le graphe")
        if "word" not in x_dict:
            raise ValueError("Aucun nœud 'word' trouvé dans le graphe")
        
        # Message passing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.n_layers - 1:
                x_dict = {k: self.activation(v) for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        if return_pooled:
            return x_dict, pooled_dict
        
        return x_dict
    
    def forward_from_pooled(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict
    ) -> dict[str, torch.Tensor]:
        """
        Forward à partir de vecteurs déjà poolés.
        """
        # Projection initiale
        x_dict = {k: self.input_proj(v) for k, v in x_dict.items()}
        
        # Message passing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.n_layers - 1:
                x_dict = {k: self.activation(v) for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        return x_dict
    
    def get_attention_weights(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Retourne les poids d'attention pour visualisation.
        """
        if self.pooling is None:
            return None
        
        if features.dim() == 2:
            return None
        
        scores = self.pooling.attention_proj(features).squeeze(-1)
        
        if attention_mask is not None:
            if attention_mask.shape[1] != scores.shape[1]:
                if attention_mask.shape[1] > scores.shape[1]:
                    attention_mask = attention_mask[:, :scores.shape[1]]
                else:
                    pad_size = scores.shape[1] - attention_mask.shape[1]
                    pad = torch.ones(attention_mask.shape[0], pad_size, 
                                    dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, pad], dim=1)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        
        weights = torch.softmax(scores, dim=-1)
        return weights
    
    def get_hidden_dim(self) -> int:
        """Retourne la dimension cachée actuelle."""
        return self.hidden_dim
