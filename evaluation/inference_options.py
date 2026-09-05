"""
Options d'inférence pour un nouveau nœud audio, jamais vu (cf. slides
"Inférence -- Option 1/2/3").

Le modèle n'est entraîné qu'en dropout SOFT (cf. graph/dropout.py) --
l'Option 1 est donc une ABLATION hors distribution, pas un scénario
d'entraînement dédié. Elle sert de borne basse de comparaison.

Toutes ces fonctions supposent un modèle déjà entraîné (checkpoint chargé),
et un graphe de référence (cf. `reference_fraction`, hyperparamètre estimé
en cross-validation) déjà construit.
"""

import torch
from models.gnn import HeterogeneousGNN
from models.link_predictor import LinkPredictor


def option1_audio_only_ablation(
    new_audio_features: torch.Tensor,
    reference_audio_graph_edges: torch.Tensor,
    gnn: HeterogeneousGNN,
    new_node_audio_audio_edges: torch.Tensor,  # ✅ Arêtes connectant le nouveau nœud
    reference_audio_features: torch.Tensor,    # ✅ Features de TOUS les nœuds de référence
) -> torch.Tensor:
    device = reference_audio_features.device
    
    # ✅ 1. Features de TOUS les nœuds audio (référence + nouveau)
    all_audio_features = torch.cat([reference_audio_features, new_audio_features], dim=0)
    
    # ✅ 2. Toutes les arêtes (référence + nouvelles)
    all_audio_audio_edges = torch.cat(
        [reference_audio_graph_edges, new_node_audio_audio_edges], dim=1
    )

    # ✅ 3. Vérifier que le GNN est initialisé
    if gnn.hidden_dim is None:
        raise RuntimeError("gnn.hidden_dim est None -- faire warmup d'abord")

    # ✅ 4. x_dict avec TOUS les nœuds
    features_dict = {
        "audio": all_audio_features,  # Tous les audios
        "word": _dummy_word_placeholder(gnn.hidden_dim, device),  # Nœud factice
    }
    
    # ✅ 5. edge_index_dict avec toutes les arêtes
    edge_index_dict = {
        ("audio", "similar_to", "audio"): all_audio_audio_edges,
        # PAS de "transcribed_as" : c'est l'ablation !
    }

    # ✅ 6. Appel correct : edge_index_dict en KEYWORD
    out = gnn(features_dict, edge_index_dict=edge_index_dict)
    
    # ✅ 7. Retourner le dernier nœud (le nouveau)
    return out["audio"][-1]


def option2_heterogeneous_reference(
    new_audio_features: torch.Tensor,
    reference_graph,  # HeteroData -- graphe hétérogène complet (audio+word+arêtes)
    gnn: HeterogeneousGNN,
    new_node_audio_audio_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Introduit le nouveau nœud dans le graphe hétérogène de référence
    (audio ET texte, avec leurs arêtes) -- bénéficie du signal indirect
    via ses voisins audio, eux-mêmes reliés à du texte (Canal 2).

    Args:
        new_node_audio_audio_edges: arêtes k-NN reliant le nouveau nœud aux
            ancres du graphe de référence (cf. build_graph.build_audio_audio_edges)
    """
    # S'assurer que new_audio_features est en 2D
    if new_audio_features.dim() == 1:
        new_audio_features = new_audio_features.unsqueeze(0)
    
    # Vérifier que le GNN est initialisé
    if gnn.hidden_dim is None:
        raise RuntimeError(
            "gnn.hidden_dim est None -- le GNN n'a jamais fait de forward "
            "réel. Appeler RetrievalEvaluator._warmup_gnn() avant toute "
            "inférence."
        )
    
    # Construire x_dict avec TOUS les nœuds
    x_dict = {
        "audio": torch.cat([reference_graph["audio"].x, new_audio_features], dim=0),
        "word": reference_graph["word"].x,
    }
    
    # Copier et mettre à jour les arêtes
    edge_index_dict = dict(reference_graph.edge_index_dict)
    
    # Ajouter les arêtes audio-audio du nouveau nœud
    existing_audio_audio = edge_index_dict[("audio", "similar_to", "audio")]
    edge_index_dict[("audio", "similar_to", "audio")] = torch.cat(
        [existing_audio_audio, new_node_audio_audio_edges], dim=1
    )
    
    # CORRECTION 1: edge_index_dict en KEYWORD, jamais positionnel
    out = gnn(x_dict, edge_index_dict=edge_index_dict)
    
    # Le nouveau nœud est en dernière position (dernier ajouté)
    return out["audio"][-1]


def option3_reconnection_via_link_predictor(
    new_audio_features: torch.Tensor,
    reference_graph,
    gnn: HeterogeneousGNN,
    link_predictor: LinkPredictor,
    new_node_audio_audio_edges: torch.Tensor,
    confidence_threshold: float = 0.6,
    top_k_candidates: int = 100,
) -> torch.Tensor:
    """
    Passe 1 (Option 2) -> requête au link predictor -> Passe 2 avec le(s)
    lien(s) prédit(s) ajouté(s) explicitement.

    Point de vigilance (cf. slide) : la qualité de la seconde représentation
    dépend de la fiabilité de la première -- d'où le seuillage de confiance.
    """
    # S'assurer que new_audio_features est en 2D
    if new_audio_features.dim() == 1:
        new_audio_features = new_audio_features.unsqueeze(0)
    
    # Vérifier que le GNN est initialisé
    if gnn.hidden_dim is None:
        raise RuntimeError(
            "gnn.hidden_dim est None -- le GNN n'a jamais fait de forward "
            "réel. Appeler RetrievalEvaluator._warmup_gnn() avant toute "
            "inférence."
        )
    
    # Passe 1 : représentation ancrée dans le graphe hétérogène (Option 2)
    h_audio_pass1 = option2_heterogeneous_reference(
        new_audio_features, reference_graph, gnn, new_node_audio_audio_edges
    )

    # Requête au link predictor pour proposer des liens texte
    top_k_indices, top_k_scores = link_predictor.propose_edges(
        h_audio_pass1, reference_graph["word"].x, top_k=top_k_candidates
        
    )

    # Seuillage de confiance : ne garder que les liens au-dessus du seuil
    confident_mask = top_k_scores > confidence_threshold
    if not confident_mask.any():
        # Aucun lien assez fiable : repli sur la représentation de la Passe 1
        return h_audio_pass1

    predicted_word_indices = top_k_indices[confident_mask]

    # CORRECTION 2: L'index du nouveau nœud est sa position RÉELLE
    # avant concaténation (dernière ligne après concaténation)
    new_node_index = reference_graph["audio"].x.shape[0]  # avant concaténation
    
    # CORRECTION 3: Utiliser new_node_index au lieu de -1
    new_cross_edges = torch.stack(
        [
            torch.full_like(predicted_word_indices, fill_value=new_node_index),
            predicted_word_indices,
        ]
    )

    # Passe 2 : ajouter le(s) lien(s) prédit(s) explicitement, refaire le forward
    x_dict = {
        "audio": torch.cat([reference_graph["audio"].x, new_audio_features], dim=0),
        "word": reference_graph["word"].x,
    }
    
    edge_index_dict = dict(reference_graph.edge_index_dict)
    
    # Ajouter les arêtes audio-audio du nouveau nœud
    edge_index_dict[("audio", "similar_to", "audio")] = torch.cat(
        [edge_index_dict[("audio", "similar_to", "audio")], new_node_audio_audio_edges], dim=1
    )
    
    # Ajouter les arêtes cross prédites (audio -> word)
    existing_cross = edge_index_dict[("audio", "transcribed_as", "word")]
    edge_index_dict[("audio", "transcribed_as", "word")] = torch.cat(
        [existing_cross, new_cross_edges], dim=1
    )
    
    # CORRECTION 4: Ajouter aussi l'arête inverse (word -> audio) si le GNN l'utilise
    # Certains GNN utilisent les relations inverses pour le message passing
    # Vérifier si la relation inverse existe dans le graphe de référence
    if ("word", "rev_transcribed_as", "audio") in edge_index_dict:
        # Ajouter l'arête inverse symétrique
        inverse_cross_edges = torch.stack([
            predicted_word_indices,
            torch.full_like(predicted_word_indices, fill_value=new_node_index)
        ])
        existing_inverse = edge_index_dict[("word", "rev_transcribed_as", "audio")]
        edge_index_dict[("word", "rev_transcribed_as", "audio")] = torch.cat(
            [existing_inverse, inverse_cross_edges], dim=1
        )

    # CORRECTION 5: edge_index_dict en KEYWORD, jamais positionnel
    out = gnn(x_dict, edge_index_dict=edge_index_dict)
    
    # Le nouveau nœud est en dernière position (dernier ajouté)
    return out["audio"][-1]
    



def _dummy_word_placeholder(hidden_dim: int, device: torch.device) -> torch.Tensor:
    """
    Nœud 'word' factice pour satisfaire l'exigence structurelle de
    gnn.py (cf. point 2 ci-dessus) -- UNIQUEMENT utilisé par l'Option 1,
    jamais relié à l'audio par aucune arête.

    IMPORTANT : hidden_dim doit correspondre à la dimension déjà établie
    par le GNN (gnn.hidden_dim, fixée au premier forward réel -- cf.
    retrieval.py, RetrievalEvaluator._warmup_gnn). Le calculer autrement
    changerait les dimensions attendues par les sous-modules déjà chargés
    depuis le checkpoint.
    """
    return torch.zeros(1, hidden_dim, device=device)
