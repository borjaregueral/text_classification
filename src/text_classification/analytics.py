"""
"""

import re
from typing import List, Dict, Tuple
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def handle_negations(text: str, negations: List[str]) -> str:
    """
    Detect and transform negated phrases in the text.
    
    Args:
    text (str): The input text.
    negations (List[str]): List of negation words.
    
    Returns:
    str: The text with negated phrases transformed.
    """
    words = text.split()  # Tokenize the text
    transformed_words = []
    negate = False

    i = 0
    while i < len(words):
        word = words[i]
        # If negation detected, append "not" to the following word
        if word in negations and i + 1 < len(words):
            transformed_words.append(f"not_{words[i + 1]}")
            negate = True
            i += 1  # Skip the next word as it's combined with negation
        elif negate:
            transformed_words.append(f"not_{word}")
            negate = False
        else:
            transformed_words.append(word)
        i += 1
    return ' '.join(transformed_words)

def handle_repeated_characters(word: str) -> str:
    """
    Handle repeated characters in a word (e.g., "soooo" -> "soo").
    
    Args:
    word (str): The input word.
    
    Returns:
    str: The word with repeated characters handled.
    """
    return re.sub(r'(.)\1+', r'\1\1', word)

def clean_text(text: str, negations: List[str]) -> str:
    """
    Preprocess the text for sentiment analysis.
    
    Args:
    text (str): The input text.
    negations (List[str]): List of negation words.
    
    Returns:
    str: The cleaned and preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove all punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Split the text into words
    words = text.split()
    
    # Remove stopwords, but keep negations and important words like "but", "very", etc.
    words = [word for word in words if word not in stop_words or word in negations]
    
    # Handle negations by concatenating them with the following word
    neg_handled_text = handle_negations(' '.join(words), negations)
    
    # Tokenize the processed text after handling negations
    words_processed = neg_handled_text.split()
    
    # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_processed]
    
    # Handle repeated characters
    lemmatized_words = [handle_repeated_characters(word) for word in lemmatized_words]
    
    # Join the cleaned words back into a string
    cleaned_text = ' '.join(lemmatized_words)
    
    return cleaned_text



def load_glove_embeddings(glove_file: str) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a file into a dictionary.
    
    Args:
        glove_file (str): Path to the GloVe file.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping words to their GloVe embeddings.
    """
    with open(glove_file, 'r', encoding='utf8') as f:
        return {
            line.split()[0]: np.asarray(line.split()[1:], dtype='float32')
            for line in f
        }

def initialize_random_embeddings(vocab_size: int, embedding_dim: int) -> np.ndarray:
    """
    Initialize random embeddings for words not found in GloVe.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
    
    Returns:
        np.ndarray: Randomly initialized embedding matrix.
    """
    return np.random.uniform(-0.05, 0.05, (vocab_size, embedding_dim))

def create_embedding_matrix(word_index: Dict[str, int], embeddings_index: Dict[str, np.ndarray], 
                            max_words: int, embedding_dim: int) -> np.ndarray:
    """
    Create an embedding matrix using GloVe embeddings and the word index from the tokenizer.
    
    Args:
        word_index (Dict[str, int]): Dictionary mapping words to their indices.
        embeddings_index (Dict[str, np.ndarray]): Dictionary mapping words to their GloVe embeddings.
        max_words (int): Maximum number of words to consider.
        embedding_dim (int): Dimension of the embeddings.
    
    Returns:
        np.ndarray: Embedding matrix.
    """
    vocab_size = min(max_words, len(word_index) + 1)
    embedding_matrix = initialize_random_embeddings(vocab_size, embedding_dim)
    
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    return embedding_matrix