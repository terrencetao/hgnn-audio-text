"""
Représentations linguistiques et similarités (cf. slides "Représentations
linguistiques" : LaBSE / sac de phonèmes / représentation articulatoire).

Ces représentations sont des fonctions DÉTERMINISTES du mot/phonème --
elles ne dépendent jamais du locuteur ni de l'audio. C'est ce qui justifie
qu'un probing "identité du locuteur" sur les nœuds G_l seul ne peut
structurellement rien donner (cf. discussion).
"""
from transformers import AutoTokenizer, AutoModel
import torch
from enum import Enum
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import panphon
import panphon.segment


class LinguisticRepresentation(str, Enum):
    LABSE = "labse"
    AFRIBERTA = "afriberta"
    ARTICULATORY = "articulatory"


# Cache global pour les modèles (évite de recharger à chaque appel)
_LABSE_MODEL = None
_AfriBerta_MODEL = None


def get_labse_model():
    """Charge et retourne le modèle LaBSE en cache."""
    global _LABSE_MODEL
    if _LABSE_MODEL is None:
        _LABSE_MODEL = SentenceTransformer('sentence-transformers/LaBSE')
    return _LABSE_MODEL



_AFRIBERTA_TOKENIZER = None
_AFRIBERTA_MODEL = None


def get_afriberta_model():
    """Charge et retourne AfriBERTa en cache."""
    global _AFRIBERTA_TOKENIZER, _AFRIBERTA_MODEL

    if _AFRIBERTA_MODEL is None:
        model_name = "castorini/afriberta_base"

        _AFRIBERTA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _AFRIBERTA_TOKENIZER.model_max_length = 512

        _AFRIBERTA_MODEL = AutoModel.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _AFRIBERTA_MODEL = _AFRIBERTA_MODEL.to(device)
        _AFRIBERTA_MODEL.eval()

    return _AFRIBERTA_TOKENIZER, _AFRIBERTA_MODEL


def compute_labse_embeddings(phrases: list[str]) -> torch.Tensor:
    """
    Encode chaque phrase avec LaBSE.

    Returns:
        Tensor de forme (N, 768).
    """
    model = get_labse_model()
    embeddings = model.encode(phrases, convert_to_tensor=True)
    return embeddings


def compute_afriberta_embeddings(
    phrases: list[str],
    batch_size: int = 16
) -> torch.Tensor:
    """
    Encode les mots/phrases avec AfriBERTa.

    Mean pooling masqué sur les tokens suivi d'une normalisation L2.

    Returns:
        Tensor de forme (N, 768).
    """
    tokenizer, model = get_afriberta_model()
    device = next(model.parameters()).device

    all_embeddings = []

    for start in range(0, len(phrases), batch_size):
        batch_phrases = phrases[start:start + batch_size]

        inputs = tokenizer(
            batch_phrases,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).to(
            hidden_states.dtype
        )

        embeddings = (
            (hidden_states * attention_mask).sum(dim=1)
            / attention_mask.sum(dim=1).clamp(min=1e-9)
        )

        embeddings = torch.nn.functional.normalize(
            embeddings, p=2, dim=-1
        )

        all_embeddings.append(embeddings.cpu())

    if not all_embeddings:
        return torch.empty((0, 768), dtype=torch.float32)

    return torch.cat(all_embeddings, dim=0)
    
def compute_afriberta_embeddings(phrases: list[str]) -> torch.Tensor:
    """
    Encode chaque phrase avec LaBSE (Feng et al., 2022).
    
    Args:
        phrases: Liste de phrases (ex: ["hello world", "bonjour le monde"])
    
    Returns:
        embeddings: (N_phrases, 768) - Embeddings LaBSE pour chaque phrase
    """
    model = get_afriberta_model()
    # LaBSE encode directement des phrases complètes
    embeddings = model.encode(phrases, convert_to_tensor=True)
    return embeddings


def compute_bag_of_phonemes(
    phrases_phonemes: list[list[list[str]]], 
    phoneme_vocab: list[str]
) -> torch.Tensor:
    """
    Vecteur binaire d'occurrence de phonèmes pour chaque phrase.
    
    Args:
        phrases_phonemes: Liste de phrases, chaque phrase est une liste de mots,
                         chaque mot est une liste de phonèmes
                         Ex: [ [['d', 'a', 'w', 'n'], ['g', 'o', 'ʊ', 'p']], ... ]
        phoneme_vocab: Vocabulaire de tous les phonèmes possibles
    
    Returns:
        vectors: (N_phrases, len(phoneme_vocab)) - Vecteur binaire par phrase
    """
    vocab_index = {p: i for i, p in enumerate(phoneme_vocab)}
    vectors = torch.zeros(len(phrases_phonemes), len(phoneme_vocab))
    
    for i, phrase in enumerate(phrases_phonemes):
        # Compter les phonèmes uniques dans toute la phrase
        unique_phonemes = set()
        for word_phonemes in phrase:
            for p in word_phonemes:
                if p in vocab_index:
                    unique_phonemes.add(p)
        
        # Mettre à 1 pour chaque phonème présent
        for p in unique_phonemes:
            vectors[i, vocab_index[p]] = 1.0
    
    return vectors


def compute_bag_of_phonemes_weighted(
    phrases_phonemes: list[list[list[str]]], 
    phoneme_vocab: list[str]
) -> torch.Tensor:
    """
    Version pondérée du sac de phonèmes (compte les occurrences).
    
    Args:
        phrases_phonemes: Liste de phrases, chaque phrase est une liste de mots,
                         chaque mot est une liste de phonèmes
        phoneme_vocab: Vocabulaire de tous les phonèmes possibles
    
    Returns:
        vectors: (N_phrases, len(phoneme_vocab)) - Vecteur de fréquences par phrase
    """
    vocab_index = {p: i for i, p in enumerate(phoneme_vocab)}
    vectors = torch.zeros(len(phrases_phonemes), len(phoneme_vocab))
    
    for i, phrase in enumerate(phrases_phonemes):
        for word_phonemes in phrase:
            for p in word_phonemes:
                if p in vocab_index:
                    vectors[i, vocab_index[p]] += 1.0
    
    return vectors


def compute_articulatory_features(
    phrases_phonemes: list[list[list[str]]],
    mode: str = 'flat'  # 'flat', 'mean', 'concat'
) -> list[torch.Tensor]:
    """
    Représentation articulatoire pour chaque phrase.
    
    Pour chaque phonème, un vecteur de traits articulatoires est extrait.
    Les traits incluent: consonne, occlusive, fricative, voisée, labiale,
    alvéolaire, voyelle, etc.
    
    Args:
        phrases_phonemes: Liste de phrases, chaque phrase est une liste de mots,
                         chaque mot est une liste de phonèmes IPA
        mode: 'flat' - aplatit tous les vecteurs
              'mean' - moyenne des vecteurs de la phrase
              'concat' - concatène les vecteurs (taille variable)
    
    Returns:
        features: Liste de tensors, un par phrase
    """
    ft = panphon.FeatureTable()
    features_list = []
    
    for phrase in phrases_phonemes:
        phrase_vectors = []
        
        for word_phonemes in phrase:
            for phoneme in word_phonemes:
                try:
                    # Extraire les traits articulatoires du phonème
                    vector = ft.word_to_vector_list([phoneme], numeric=True)
                    if vector:
                        phrase_vectors.append(np.array(vector[0]))
                except (IndexError, ValueError):
                    # Phonème non reconnu, on ignore
                    print(f"Phonème non reconnu : {phoneme}")
                    continue
        
        if not phrase_vectors:
            # Si aucun phonème n'a été reconnu, retourner un vecteur nul
            features_list.append(torch.zeros(ft.n_features))
            continue
        
        # Agréger les vecteurs selon le mode
        if mode == 'mean':
            # Moyenne des vecteurs de tous les phonèmes de la phrase
            aggregated = np.mean(phrase_vectors, axis=0)
        elif mode == 'concat':
            # Concaténation de tous les vecteurs (taille variable)
            aggregated = np.concatenate(phrase_vectors)
        else:  # 'flat' par défaut
            # Aplatir tous les vecteurs (taille variable)
            aggregated = np.concatenate(phrase_vectors)
        
        features_list.append(torch.tensor(aggregated, dtype=torch.float))
    
    return features_list


def compute_articulatory_features_fixed_size(
    phrases_phonemes: list[list[list[str]]],
    max_phonemes_per_phrase: int = 50
) -> torch.Tensor:
    """
    Version avec taille fixe pour le batch processing.
    Pad ou tronque les séquences à max_phonemes_per_phrase.
    
    Args:
        phrases_phonemes: Liste de phrases, chaque phrase est une liste de mots,
                         chaque mot est une liste de phonèmes IPA
        max_phonemes_per_phrase: Nombre maximum de phonèmes par phrase
    
    Returns:
        features: (N_phrases, max_phonemes_per_phrase * n_features)
    """
    ft = panphon.FeatureTable()
    n_features = ft.n_features
    all_features = []
    
    for phrase in phrases_phonemes:
        phrase_vectors = []
        
        for word_phonemes in phrase:
            for phoneme in word_phonemes:
                try:
                    vector = ft.word_to_vector_list([phoneme], numeric=True)
                    if vector:
                        phrase_vectors.append(np.array(vector[0]))
                except (IndexError, ValueError):
                    continue
        
        # Pad ou tronque
        if len(phrase_vectors) > max_phonemes_per_phrase:
            phrase_vectors = phrase_vectors[:max_phonemes_per_phrase]
        else:
            # Pad avec des zéros
            while len(phrase_vectors) < max_phonemes_per_phrase:
                phrase_vectors.append(np.zeros(n_features))
        
        # Aplatir
        flat_vector = np.concatenate(phrase_vectors)
        all_features.append(torch.tensor(flat_vector, dtype=torch.float))
    
    return torch.stack(all_features)


def linguistic_similarity(features: torch.Tensor) -> torch.Tensor:
    """
    Matrice de similarité cosinus (N_words, N_words), utilisée à la fois
    pour construire les arêtes G_l <-> G_l et les arêtes croisées
    secondaires (mots proches de la transcription exacte).
    """
    normed = torch.nn.functional.normalize(features, dim=-1)
    return normed @ normed.T
    
    

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de phrases
    phrases = [
        "hello world",
        "bonjour le monde",
        "good morning"
    ]
    
    # Exemple de segmentation phonémique (simplifiée)
    # Dans la réalité, utilisez un outil comme g2p ou espeak
    phrases_phonemes = [
        [['h', 'ə', 'l', 'o', 'ʊ'], ['w', 'ɜ', 'r', 'l', 'd']],
        [['b', 'ɔ', 'ʒ', 'u', 'r'], ['l', 'ə'], ['m', 'ɔ', 'd']],
        [['g', 'ʊ', 'd'], ['m', 'ɔ', 'r', 'n', 'ɪ', 'ŋ']]
    ]
    
    phoneme_vocab = ['h', 'ə', 'l', 'o', 'ʊ', 'w', 'ɜ', 'r', 'd', 'b', 'ɔ', 'ʒ', 'u', 'm', 'g', 'n', 'ɪ', 'ŋ']
    
    # Test LaBSE
    sim_labse = compute_linguistic_similarity(
        phrases,
        LinguisticRepresentation.LABSE
    )
    print(f"Similarité LaBSE:\n{sim_labse}\n")
    
    # Test BAG_OF_PHONEMES
    sim_bop = compute_linguistic_similarity(
        phrases,
        LinguisticRepresentation.BAG_OF_PHONEMES,
        phoneme_vocab=phoneme_vocab,
        phrases_phonemes=phrases_phonemes
    )
    print(f"Similarité Sac de Phonèmes:\n{sim_bop}\n")
    
    # Test ARTICULATORY
    sim_art = compute_linguistic_similarity(
        phrases,
        LinguisticRepresentation.ARTICULATORY,
        phrases_phonemes=phrases_phonemes
    )
    print(f"Similarité Articulatoire:\n{sim_art}\n")





