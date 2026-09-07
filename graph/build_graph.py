"""
Construction du graphe hétérogène (G_a, G_l, arêtes croisées).

Rappel de la structure:
- G_a : nœuds audio, connectés par similarité acoustique avec seuil + MST (netbone)
- G_l : nœuds linguistiques, connectés entre eux si sim_l(v_i, v_j) > threshold
- Arêtes croisées G_a <-> G_l : la transcription exacte (poids fort) ET les
  nœuds linguistiques proches de cette transcription (poids = similarité)
"""

from dataclasses import dataclass
import torch
import numpy as np
from torch_geometric.data import HeteroData
import networkx as nx
import netbone as nb
from netbone.filters import boolean_filter, threshold_filter, fraction_filter
import torch.nn.functional as F
import librosa


@dataclass
class GraphBuildConfig:
    similarity_threshold: float = 0.6  # Seuil partagé pour audio-audio et linguistique
    use_mst: bool = True  # Appliquer le Maximum Spanning Tree avec netbone
    backbone_method: str = "mst"  # "mst", "disparity", "noise_corrected", etc.
    ensure_connectivity: bool = True  # Assurer que le graphe audio est connexe


def compute_cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Calcule la matrice de similarité cosinus entre toutes les paires.
    
    Args:
        features: (N, D)
    
    Returns:
        sim_matrix: (N, N) avec sim[i, j] = cos_sim(features[i], features[j])
    """
    features_norm = features / features.norm(dim=1, keepdim=True)
    return features_norm @ features_norm.T
    



import torch
import torch.nn.functional as F
import numpy as np
import librosa


def compute_cosine_similarity_matrix(
    features: torch.Tensor
) -> torch.Tensor:
    """
    Similarité cosine classique pour des vecteurs (N, D).

    La représentation linguistique (LaBSE, AfriBERTa, etc.) est calculée
    en amont ; cette fonction reste indépendante du modèle utilisé.
    """

    if features.dim() != 2:
        raise ValueError(
            f"Expected (N, D), got {features.shape}"
        )

    features = F.normalize(
        features,
        p=2,
        dim=-1
    )

    return features @ features.T


def dtw_cosine_distance(
    x: torch.Tensor,
    y: torch.Tensor
) -> float:

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # Matrice frame-frame
    cosine_similarity = x @ y.T

    # cosine distance
    cost = 1.0 - cosine_similarity

    cost = cost.detach().cpu().numpy()

    # DTW
    accumulated_cost, path = librosa.sequence.dtw(
        C=cost,
        backtrack=True
    )

    # Normalisation par la longueur du chemin
    return float(
        accumulated_cost[-1, -1] / len(path)
    )


def compute_audio_dtw_similarity_matrix(
    frame_features: torch.Tensor
) -> torch.Tensor:

    """
    frame_features : (N, T, D)

    Retourne une matrice de similarité (N, N).
    """

    if frame_features.dim() != 3:
        raise ValueError(
            f"Expected (N, T, D), got {frame_features.shape}"
        )

    N = frame_features.shape[0]

    distance_matrix = torch.zeros(
        (N, N),
        dtype=torch.float32
    )

    for i in range(N):

        for j in range(i, N):

            distance = dtw_cosine_distance(
                frame_features[i],
                frame_features[j]
            )

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    # Distance -> similarity
    similarity_matrix = torch.exp(
        -distance_matrix
    )

    return similarity_matrix

def apply_netbone_backbone(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    method: str = "mst",
    ensure_connectivity: bool = True,
    sim_matrix: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applique le backbone filtering avec netbone sur le graphe.
    
    Utilise netbone.maximum_spanning_tree pour le Maximum Spanning Tree.
    
    Args:
        edge_index: (2, E)
        edge_weight: (E,)
        num_nodes: nombre total de nœuds
        method: méthode de backbone à utiliser
        ensure_connectivity: assurer que le graphe résultat est connexe
        sim_matrix: (N, N) -- FIX (cf. discussion) : matrice de similarité
            complète, utilisée pour choisir une VRAIE meilleure paire lors
            du pontage de composantes déconnectées (voir point 2 ci-dessous).
            Optionnelle pour compatibilité ascendante ; sans elle, on
            retombe sur l'ancien comportement (poids arbitraire 0.1).
    
    Returns:
        filtered_edge_index: (2, E')
        filtered_edge_weight: (E',)
    """
    if edge_index.shape[1] == 0 or num_nodes <= 1:
        return edge_index, edge_weight
    
    # Créer un graphe NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Ajouter les arêtes avec leurs poids
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        w = edge_weight[i].item()
        G.add_edge(u, v, weight=w)
    
    # Si le graphe n'est pas connexe et qu'on veut assurer la connectivité
    if ensure_connectivity and not nx.is_connected(G):
        # Ajouter des arêtes pour connecter les composantes
        components = list(nx.connected_components(G))
        
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            best_weight = -1
            best_edge = None
            
            for u in comp1:
                for v in comp2:
                    # FIX (cf. discussion, point 2) : u et v appartiennent
                    # par définition à des composantes DIFFÉRENTES, donc
                    # G.has_edge(u, v) est TOUJOURS faux ici -- l'ancienne
                    # branche qui le vérifiait était du code mort, et `w`
                    # valait systématiquement 0.1 (poids arbitraire), donc
                    # `best_edge` se figeait sur la PREMIÈRE paire
                    # rencontrée, pas la meilleure. On utilise maintenant
                    # la vraie similarité acoustique quand elle est
                    # disponible, pour choisir un pont pertinent plutôt
                    # qu'arbitraire entre les deux composantes.
                    if sim_matrix is not None:
                        w = sim_matrix[u, v].item()
                    else:
                        w = 0.1  # repli si sim_matrix non fournie (compat. ascendante)
                    
                    if w > best_weight:
                        best_weight = w
                        best_edge = (u, v)
            
            if best_edge is not None:
                G.add_edge(best_edge[0], best_edge[1], weight=best_weight)
    
    # Si le graphe est vide, retourner des tensors vides
    if G.number_of_edges() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
    
    # Appliquer le backbone filtering avec netbone
    #start_time = time.time()
    
    if method == "mst":
        # Maximum Spanning Tree avec netbone
        backbone = nb.maximum_spanning_tree(G)
        
    elif method == "disparity":
        # Disparity Filter - nécessite des poids normalisés
        for node in G.nodes():
            incident_edges = list(G.edges(node, data=True))
            total_weight = sum([data['weight'] for _, _, data in incident_edges])
            if total_weight > 0:
                for u, v, data in incident_edges:
                    data['normalized_weight'] = data['weight'] / total_weight
        
        backbone = nb.DisparityFilter(G, weight='weight', alpha=0.05)
        
    elif method == "noise_corrected":
        backbone = nb.NoiseCorrected(G, weight='weight', method='poisson')
        
    elif method == "global_threshold":
        backbone = nb.GlobalThreshold(G, weight='weight', threshold=0.5)
        
    elif method == "sco":
        backbone = nb.SCO(G, weight='weight', alpha=0.05)
        
    else:
        raise ValueError(f"Méthode {method} non supportée par netbone")
    
    # Filtrer les arêtes avec poids > 0 (comme dans ton code)
    backbone_filtered = boolean_filter(backbone)
    
    #elapsed_time = time.time() - start_time
    #formatted_time = str(timedelta(seconds=int(elapsed_time)))
    #print(f"   Temps de filtrage (Backbone): {formatted_time}")
    
    # Extraire les arêtes du backbone filtré
    filtered_edges = list(backbone_filtered.edges(data=True))
    
    if not filtered_edges:
        print(f"   ⚠️ Aucune arête après filtrage, utilisation du graphe original")
        # Si le filtrage supprime tout, on garde les arêtes originales avec le plus grand poids
        # Prendre les arêtes avec les plus grands poids (top 10% ou minimum n_edges)
        sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
        n_keep = max(1, len(sorted_edges) // 10)  # Garder 10% des arêtes
        filtered_edges = sorted_edges[:n_keep]
    
    # Convertir en tensors
    filtered_edge_index = torch.tensor(
        [[u, v] for u, v, _ in filtered_edges],
        dtype=torch.long
    ).T
    
    filtered_edge_weight = torch.tensor(
        [data.get('weight', 0.0) for _, _, data in filtered_edges],
        dtype=torch.float
    )
    
    print(f"   Arêtes avant filtrage: {edge_index.shape[1]}")
    print(f"   Arêtes après filtrage: {filtered_edge_index.shape[1]}")
    
    # FIX CRITIQUE (cf. discussion, point 1) : networkx dédoublonne les
    # arêtes non-dirigées -- G.edges() ne retourne chaque paire (u,v)
    # qu'UNE SEULE FOIS, alors que edge_index en entrée contenait les DEUX
    # sens (matrice de similarité symétrique). Sans cette symétrisation,
    # le message passing PyG sur ("audio","similar_to","audio") devient
    # directionnel après filtrage MST (use_mst=True par défaut) alors que
    # la relation est censée être non-dirigée.
    filtered_edge_index = torch.cat([filtered_edge_index, filtered_edge_index.flip(0)], dim=1)
    filtered_edge_weight = torch.cat([filtered_edge_weight, filtered_edge_weight])
    
    return filtered_edge_index, filtered_edge_weight


def build_audio_audio_edges(
    audio_features: torch.Tensor,
    audio_transcription_ids: torch.Tensor,
    threshold: float,
    use_mst: bool = True,
    backbone_method: str = "mst",
    ensure_connectivity: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construit les arêtes audio-audio selon la nouvelle approche :
    - Même transcription → poids = 1.0
    - Transcription différente → poids = similarité acoustique si > threshold
    - Filtrage par Maximum Spanning Tree avec netbone
    
    Args:
        audio_features: (N_audio, D) -- features acoustiques
        audio_transcription_ids: (N_audio,) -- id de la transcription pour chaque nœud
        threshold: seuil de similarité partagé
        use_mst: appliquer ou non le MST avec netbone
        backbone_method: méthode netbone à utiliser
        ensure_connectivity: assurer la connectivité du graphe
    
    Returns:
        edge_index: (2, N_edges)
        edge_weight: (N_edges,)
    """
    N = audio_features.shape[0]
    
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
    
    # Matrice de similarité acoustique
    sim_matrix = compute_audio_dtw_similarity_matrix(
    audio_features
)
    
    # Créer le masque des arêtes
    # 1. Même transcription → arête avec poids 1.0
    same_trans = audio_transcription_ids[:, None] == audio_transcription_ids[None, :]
    # Enlever la diagonale (pas d'auto-boucle)
    same_trans.fill_diagonal_(False)
    
    # 2. Transcriptions différentes avec similarité > threshold
    diff_trans = ~same_trans
    # Enlever la diagonale
    diff_trans.fill_diagonal_(False)
    # Appliquer le seuil partagé
    mask_diff = diff_trans & (sim_matrix > threshold)
    
    # Combiner les masques
    mask = same_trans | mask_diff
    
    # Extraire les arêtes
    edge_index = torch.nonzero(mask, as_tuple=False).T  # (2, N_edges)
    
    if edge_index.shape[1] == 0:
        return edge_index, torch.empty(0, dtype=torch.float)
    
    # Assigner les poids
    edge_weight = torch.zeros(edge_index.shape[1])
    row, col = edge_index
    for idx in range(edge_index.shape[1]):
        i, j = row[idx].item(), col[idx].item()
        if same_trans[i, j]:
            edge_weight[idx] = 1.0
        else:
            edge_weight[idx] = sim_matrix[i, j].item()
    
    # Appliquer le Maximum Spanning Tree avec netbone si demandé
    if use_mst and edge_index.shape[1] > 0:
        edge_index, edge_weight = apply_netbone_backbone(
            edge_index, 
            edge_weight, 
            N,
            method=backbone_method,
            ensure_connectivity=ensure_connectivity,
            sim_matrix=sim_matrix,  # FIX (cf. discussion, point 2)
        )
    
    return edge_index, edge_weight


def build_linguistic_edges(
    linguistic_features: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    Construit les arêtes G_l <-> G_l : une arête entre v_i et v_j ssi
    sim_l(v_i, v_j) > threshold (même seuil que pour audio).
    
    Args:
        linguistic_features: (N_words, D_l)
        threshold: seuil de similarité partagé
    
    Returns:
        edge_index: (2, N_edges)
    """
    N = linguistic_features.shape[0]
    
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    
    sim_matrix = compute_cosine_similarity_matrix(linguistic_features)
    
    # Masque pour les arêtes > seuil (en excluant la diagonale)
    mask = sim_matrix > threshold
    mask.fill_diagonal_(False)
    
    edge_index = torch.nonzero(mask, as_tuple=False).T
    
    return edge_index


def build_cross_edges(
    audio_transcription_ids: torch.Tensor,
    linguistic_features: torch.Tensor,
    threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construit TOUTES les arêtes G_a <-> G_l pour chaque nœud audio :
    - une arête de poids 1 vers sa transcription exacte
    - une arête de poids sim_l(...) vers chaque nœud linguistique proche
    
    Args:
        audio_transcription_ids: (N_audio,) -- index du mot transcrit
        linguistic_features: (N_words, D_l)
        threshold: seuil de similarité partagé
    
    Returns:
        edge_index: (2, N_cross_edges)
        edge_weight: (N_cross_edges,)
    """
    N_audio = audio_transcription_ids.shape[0]
    N_words = linguistic_features.shape[0]
    
    if N_audio == 0 or N_words == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
    
    # Matrice de similarité linguistique
    sim_matrix = compute_cosine_similarity_matrix(linguistic_features)
    
    edges_list = []
    weights_list = []
    
    for i, trans_id in enumerate(audio_transcription_ids):
        # Arête vers la transcription exacte (toujours présente)
        edges_list.append([i, trans_id.item()])
        weights_list.append(1.0)
        
        # Arêtes vers les mots proches (avec le même seuil)
        sim_row = sim_matrix[trans_id]
        for w in range(N_words):
            if w != trans_id and sim_row[w] > threshold:
                edges_list.append([i, w])
                weights_list.append(sim_row[w].item())
    
    if not edges_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
    
    edge_index = torch.tensor(edges_list, dtype=torch.long).T
    edge_weight = torch.tensor(weights_list, dtype=torch.float)
    
    return edge_index, edge_weight


def build_heterogeneous_graph(
    audio_features: torch.Tensor,
    linguistic_features: torch.Tensor,
    audio_transcription_ids: torch.Tensor,
    config: GraphBuildConfig,
) -> HeteroData:
    """
    Assemble le graphe complet G = (G_a, G_l, arêtes croisées).
    Tous les seuils sont partagés via config.similarity_threshold.
    """
    data = HeteroData()
    
    data["audio"].x = audio_features
    data["word"].x = linguistic_features
    
    # Arêtes audio-audio (avec seuil partagé + MST netbone)
    edge_audio_audio, weight_audio_audio = build_audio_audio_edges(
        audio_features,
        audio_transcription_ids,
        threshold=config.similarity_threshold,
        use_mst=config.use_mst,
        backbone_method=getattr(config, 'backbone_method', 'mst'),
        ensure_connectivity=getattr(config, 'ensure_connectivity', True)
    )
    data["audio", "similar_to", "audio"].edge_index = edge_audio_audio
    data["audio", "similar_to", "audio"].edge_weight = weight_audio_audio
    
    # Arêtes linguistiques (avec le même seuil partagé)
    edge_word_word = build_linguistic_edges(
        linguistic_features,
        threshold=config.similarity_threshold
    )
    data["word", "similar_to", "word"].edge_index = edge_word_word
    
    # Arêtes croisées (avec le même seuil partagé)
    edge_cross, weight_cross = build_cross_edges(
        audio_transcription_ids,
        linguistic_features,
        threshold=config.similarity_threshold
    )
    data["audio", "transcribed_as", "word"].edge_index = edge_cross
    data["audio", "transcribed_as", "word"].edge_weight = weight_cross
    
    # Arête inverse pour message passing bidirectionnel
    if edge_cross.shape[1] > 0:
        data["word", "rev_transcribed_as", "audio"].edge_index = edge_cross.flip(0)
        data["word", "rev_transcribed_as", "audio"].edge_weight = weight_cross
    
    return data

