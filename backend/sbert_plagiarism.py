import json
import sys
import time
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Levenshtein import distance
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import psutil
import os
import multiprocessing
import re
from statistics import median
from math import ceil
from fuzzywuzzy import fuzz

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)
sys.stderr.flush()

# Log memory usage
def log_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

# Download NLTK data
logger.info("Checking NLTK resources")
log_memory()
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    logger.info("NLTK resources found")
except LookupError:
    logger.info("Downloading NLTK punkt and punkt_tab")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    logger.info("NLTK resources downloaded")

# Load SBERT model
logger.info("Initializing SBERT model: all-MiniLM-L6-v2")
start_time = time.time()
try:
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    model_path = os.path.join(cache_dir, 'models--sentence-transformers--all-MiniLM-L6-v2')
    if os.path.exists(model_path):
        logger.info(f"Model found in cache: {model_path}")
    else:
        logger.info("Model not in cache, downloading...")
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',  # Switch to 'all-MiniLM-L12-v2' if needed
        device='cpu',  # Use 'mps' for Apple Silicon if available
        cache_folder=cache_dir
    )
    end_time = time.time()
    logger.info(f"SBERT model loaded. Time: {(end_time - start_time) * 1000:.2f}ms")
    log_memory()
except Exception as e:
    logger.error(f"Failed to load SBERT model: {str(e)}")
    sys.exit(1)

def preprocess_text(text, for_sbert=False, for_edit_distance=False):
    """Clean text with mode-specific preprocessing."""
    # Normalize bracketed text and citations
    text = re.sub(r'(\w+)\[(\w+)\]', r'\1\2', text)  # e.g., demonstrat[ing] -> demonstrating
    text = re.sub(r'\[\d+\]', '', text)  # Remove [71]
    text = text.strip()
    if for_sbert:
        # Preserve quotes, punctuation
        text = re.sub(r'\s+', ' ', text)
    elif for_edit_distance:
        # Minimal cleaning
        text = re.sub(r'\s+', ' ', text)
    else:
        # Moderate cleaning for TF-IDF and n-grams
        text = text.lower()
        text = re.sub(r'[^\w\s"\'-]', ' ', text)  # Preserve quotes, apostrophes, hyphens
        text = re.sub(r'\s+', ' ', text)
    return text

def analyze_corpus(corpus, input_text):
    """Analyze corpus and input for dynamic parameters."""
    logger.info("Analyzing corpus characteristics")
    start_time = time.time()
    try:
        clean_corpus = preprocess_text(corpus, for_sbert=True)
        
        # Vocabulary size for max_features
        vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        vectorizer.fit([clean_corpus])
        vocab_size = len(vectorizer.vocabulary_)
        max_features = min(int(0.95 * vocab_size), 35000)  # Increased
        logger.info(f"Vocabulary size: {vocab_size}, max_features: {max_features}")
        
        # Tokenization pattern
        tokens = word_tokenize(clean_corpus)
        token_lengths = [len(token) for token in tokens]
        avg_token_len = sum(token_lengths) / len(token_lengths) if token_lengths else 4
        token_pattern = r'(?u)\b\w+\b'
        logger.info(f"Average token length: {avg_token_len:.2f}")
        
        # N-gram size
        input_tokens = len(word_tokenize(preprocess_text(input_text, for_sbert=True)))
        n_gram_size = 3 if len(input_text) > 100 else 2
        logger.info(f"Input tokens: {input_tokens}, n-gram size: {n_gram_size}")
        
        # Segment parameters
        sentences = sent_tokenize(clean_corpus)
        sentence_lengths = [len(sent) for sent in sentences]
        max_len = max(200, min(500, int(2 * len(input_text))))
        overlap = max(100, int(0.5 * max_len))
        logger.info(f"Median sentence length: {median(sentence_lengths) if sentence_lengths else 0}, max_len: {max_len}, overlap: {overlap}")
        
        end_time = time.time()
        logger.info(f"Corpus analysis completed. Time: {(end_time - start_time) * 1000:.2f}ms")
        return {
            'max_features': max_features,
            'token_pattern': token_pattern,
            'n_gram_size': n_gram_size,
            'max_len': max_len,
            'overlap': overlap
        }
    except Exception as e:
        logger.error(f"Corpus analysis failed: {str(e)}")
        return {
            'max_features': 20000,
            'token_pattern': r'(?u)\b\w+\b',
            'n_gram_size': 2,
            'max_len': 200,
            'overlap': 100
        }

def custom_tokenize(text, token_pattern=r'(?u)\b\w+\b'):
    """Custom tokenizer for TF-IDF and n-grams."""
    try:
        text = preprocess_text(text, for_sbert=True)  # Align with SBERT
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if len(token) > 1]  # Minimal filtering
        logger.info(f"Tokenized sample: {tokens[:5]}")
        return tokens
    except Exception as e:
        logger.error(f"Custom tokenization failed: {str(e)}, falling back to word_tokenize")
        return word_tokenize(preprocess_text(text))

def split_into_segments(reference_text, max_len=200, overlap=100):
    """Split reference into overlapping segments."""
    reference_text = preprocess_text(reference_text, for_sbert=True)
    sentences = sent_tokenize(reference_text)
    segments = []
    current_segment = ""
    for sentence in sentences:
        if len(current_segment) + len(sentence) <= max_len:
            current_segment += " " + sentence
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence
            if len(segments) > 0:
                overlap_text = segments[-1][-overlap:] + " " + sentence
                if len(overlap_text) <= max_len:
                    segments.append(overlap_text.strip())
    if current_segment:
        segments.append(current_segment.strip())
    return segments if segments else [reference_text]

def find_most_similar_segment(input_text, reference_text, max_len=200, overlap=100):
    """Find the most similar segment using SBERT, with near-exact match check."""
    try:
        input_clean = preprocess_text(input_text, for_sbert=True)
        reference_clean = preprocess_text(reference_text, for_sbert=True)
        
        # Near-exact match check
        input_len = len(input_clean)
        dist = distance(input_clean, reference_clean[:input_len + 10])
        if dist / input_len < 0.15:
            logger.info("Near-exact match found in reference text")
            return input_clean, 0.95  # High similarity for near-exact

        segments = split_into_segments(reference_text, max_len, overlap)
        if not segments:
            return reference_text, 0.0
        
        segments_clean = [preprocess_text(seg, for_sbert=True) for seg in segments]
        embeddings = model.encode([input_clean] + segments_clean, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy()[0]
        max_idx = np.argmax(similarities)
        similarity = similarities[max_idx]
        if dist / input_len < 0.3:
            similarity = min(similarity * 1.3, 1.0)  # Increased boost
        return segments[max_idx], similarity
    except Exception as e:
        logger.error(f"Failed to find most similar segment: {str(e)}")
        return reference_text, 0.0

def vector_based_comparison(input_text, reference_text, most_similar_text, max_features=35000, token_pattern=r'(?u)\b\w+\b'):
    logger.info("Starting vector-based comparison")
    start_time = time.time()
    try:
        input_text = preprocess_text(input_text, for_sbert=True)
        reference_text = preprocess_text(most_similar_text, for_sbert=True)
        logger.info(f"Input sample: {input_text[:50]}...")
        logger.info(f"Reference sample: {reference_text[:50]}...")
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None, token_pattern=token_pattern, min_df=1, tokenizer=custom_tokenize)
        tfidf_matrix = vectorizer.fit_transform([input_text, reference_text])
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        similarity = min(max(similarity, 0.0), 1.0)
        end_time = time.time()
        logger.info(f"Vector-based comparison completed. Time: {(end_time - start_time) * 1000:.2f}ms, Similarity: {similarity:.4f}")
        return {
            "similarity": float(similarity),
            "most_similar_text": most_similar_text[:100]
        }
    except Exception as e:
        logger.error(f"Vector-based comparison failed: {str(e)}")
        return {"similarity": 0.0, "most_similar_text": most_similar_text[:100]}

def edit_distance_analysis(input_text, reference_text, most_similar_text):
    logger.info("Starting edit distance analysis")
    start_time = time.time()
    try:
        input_text = preprocess_text(input_text, for_edit_distance=True)
        reference_text = preprocess_text(most_similar_text, for_edit_distance=True)
        logger.info(f"Input sample: {input_text[:50]}...")
        logger.info(f"Reference sample: {reference_text[:50]}...")
        dist = distance(input_text, reference_text)
        max_len = max(len(input_text), len(reference_text))
        similarity = max(0, 1 - (dist / max_len * min(1, 100 / max_len)))
        similarity = min(max(similarity, 0.0), 1.0)
        end_time = time.time()
        logger.info(f"Edit distance analysis completed. Time: {(end_time - start_time) * 1000:.2f}ms, Similarity: {similarity:.4f}")
        return {
            "similarity": float(similarity),
            "most_similar_text": most_similar_text[:100]
        }
    except Exception as e:
        logger.error(f"Edit distance analysis failed: {str(e)}")
        return {"similarity": 0.0, "most_similar_text": most_similar_text[:100]}

def ngram_overlap_analysis(input_text, reference_text, most_similar_text, n=3, token_pattern=r'(?u)\b\w+\b'):
    logger.info("Starting n-gram overlap analysis")
    start_time = time.time()
    def get_ngrams(text, n):
        try:
            tokens = custom_tokenize(text, token_pattern)
            logger.info(f"Tokens sample: {tokens[:5]}")
            ngram_set = set(' '.join(gram) for gram in ngrams(tokens, n))
            return ngram_set
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return set()
    
    try:
        # Hybrid n-gram (n=2 and n=3)
        input_ngrams_2 = get_ngrams(input_text, 2) if n >= 2 else set()
        input_ngrams_3 = get_ngrams(input_text, 3) if n >= 3 else set()
        ref_ngrams_2 = get_ngrams(most_similar_text, 2) if n >= 2 else set()
        ref_ngrams_3 = get_ngrams(most_similar_text, 3) if n >= 3 else set()
        input_ngrams = input_ngrams_2 | input_ngrams_3
        ref_ngrams = ref_ngrams_2 | ref_ngrams_3
        
        # Fuzzy matching for n-grams
        fuzzy_score = 0
        for input_gram in input_ngrams:
            best_match = max((fuzz.ratio(input_gram, ref_gram) / 100 for ref_gram in ref_ngrams), default=0)
            fuzzy_score += best_match
        fuzzy_similarity = fuzzy_score / len(input_ngrams) if input_ngrams else 0
        
        # Jaccard similarity
        intersection = len(input_ngrams & ref_ngrams)
        union = len(input_ngrams | ref_ngrams)
        jaccard_similarity = (intersection / union) * 1.5 if union > 0 else 0
        
        # Combine Jaccard and fuzzy
        similarity = 0.7 * jaccard_similarity + 0.3 * fuzzy_similarity
        similarity = min(max(similarity, 0.0), 1.0)
        
        logger.info(f"Input n-grams sample: {list(input_ngrams)[:5]}")
        logger.info(f"Reference n-grams sample: {list(ref_ngrams)[:5]}")
        logger.info(f"N-gram overlap analysis completed. Time: {(end_time - start_time) * 1000:.2f}ms, Similarity: {similarity:.4f}")
        return {
            "similarity": float(similarity),
            "most_similar_text": most_similar_text[:100]
        }
    except Exception as e:
        logger.error(f"N-gram overlap analysis failed: {str(e)}")
        return {"similarity": 0.0, "most_similar_text": most_similar_text[:100]}

def semantic_analysis(input_text, reference_text, most_similar_text):
    logger.info("Starting semantic analysis")
    start_time = time.time()
    try:
        input_text = preprocess_text(input_text, for_sbert=True)
        reference_text = preprocess_text(most_similar_text, for_sbert=True)
        logger.info(f"Input sample: {input_text[:50]}...")
        logger.info(f"Reference sample: {reference_text[:50]}...")
        embeddings = model.encode([input_text, reference_text], convert_to_tensor=True, show_progress_bar=False)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).cpu().numpy()[0][0]
        dist = distance(input_text, reference_text)
        if dist / len(input_text) < 0.3:
            similarity = min(similarity * 1.3, 1.0)  # Increased boost
        similarity = min(max(similarity, 0.0), 1.0)
        end_time = time.time()
        logger.info(f"Semantic analysis completed. Time: {(end_time - start_time) * 1000:.2f}ms, Similarity: {similarity:.4f}")
        return {
            "similarity": float(similarity),
            "most_similar_text": most_similar_text[:100]
        }
    except Exception as e:
        logger.error(f"Semantic analysis failed: {str(e)}")
        return {"similarity": 0.0, "most_similar_text": most_similar_text[:100]}

def check_plagiarism(input_text, reference_text):
    logger.info("Starting plagiarism check")
    start_time = time.time()
    try:
        input_text = input_text[:1000]
        if not input_text or not reference_text:
            logger.error("Invalid input: empty text or corpus")
            raise ValueError("Input text or reference corpus is empty")

        corpus_params = analyze_corpus(reference_text, input_text)
        max_features = corpus_params['max_features']
        token_pattern = corpus_params['token_pattern']
        n_gram_size = corpus_params['n_gram_size']
        max_len = corpus_params['max_len']
        overlap = corpus_params['overlap']

        most_similar_text, segment_similarity = find_most_similar_segment(input_text, reference_text, max_len, overlap)
        logger.info(f"Most similar segment: {most_similar_text[:100]}... (similarity: {segment_similarity:.4f})")

        vector_result = vector_based_comparison(input_text, reference_text, most_similar_text, max_features, token_pattern)
        edit_result = edit_distance_analysis(input_text, reference_text, most_similar_text)
        ngram_result = ngram_overlap_analysis(input_text, reference_text, most_similar_text, n_gram_size, token_pattern)
        semantic_result = semantic_analysis(input_text, reference_text, most_similar_text)

        weights = {
            "vector": 0.25,
            "edit": 0.10,
            "ngram": 0.30,
            "semantic": 0.35
        }
        logger.info(f"Weights: {weights}")

        final_similarity = (
            weights["vector"] * vector_result["similarity"] +
            weights["edit"] * edit_result["similarity"] +
            weights["ngram"] * ngram_result["similarity"] +
            weights["semantic"] * semantic_result["similarity"]
        )
        is_plagiarized = final_similarity >= 0.7

        end_time = time.time()
        logger.info(f"Plagiarism check completed. Time: {(end_time - start_time) * 1000:.2f}ms, Final similarity: {final_similarity:.4f}")
        return {
            "isPlagiarized": is_plagiarized,
            "similarityScore": final_similarity,
            "mostSimilarText": most_similar_text[:100],
            "details": {
                "vectorBased": vector_result,
                "editDistance": edit_result,
                "ngramOverlap": ngram_result,
                "semanticAnalysis": semantic_result
            }
        }
    except Exception as e:
        logger.error(f"Plagiarism check failed: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Python script started")
    start_time = time.time()
    try:
        input_text = sys.argv[1]
        reference_texts = json.loads(sys.argv[2])
        logger.info(f"Input text: {input_text[:100]}... (length: {len(input_text)})")
        logger.info(f"Reference texts: {len(reference_texts)}")
        log_memory()
        result = check_plagiarism(input_text, reference_texts[0] if reference_texts else "")
        print(json.dumps(result))
        sys.stdout.flush()
        end_time = time.time()
        logger.info(f"Script completed successfully. Time: {(end_time - start_time) * 1000:.2f}ms")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)
    finally:
        multiprocessing.current_process().close()