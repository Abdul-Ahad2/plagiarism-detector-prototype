from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def check_plagiarism(input_text, reference_text, threshold=0.8):
    """
    Check if input_text is a paraphrase of reference_text using SBERT.
    
    Args:
        input_text (str): The text to check for plagiarism.
        reference_text (str): The original reference text.
        threshold (float): Similarity score above which text is considered plagiarized (0-1).
    
    Returns:
        bool: True if plagiarism detected, False otherwise.
        float: Cosine similarity score.
    """
    # Encode the texts into embeddings
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    reference_embedding = model.encode(reference_text, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.cos_sim(input_embedding, reference_embedding).item()
    
    # Check if similarity exceeds the threshold
    is_plagiarized = similarity >= threshold
    
    return is_plagiarized, similarity

# Example usage
input_text = "One of the arm's three long bones is the humerus.  It connects to the scapula at the shoulder joint and to the ulna and radius, the other long bones of the arm, at the elbow joint. The elbow is a complicated hinge joint that connects the extremities of the ulna and radius to the humerus."
reference_text = "The humerus is one of the three long bones of the arm. It joins with the scapula at the shoulder joint and with the other long bones of the arm, the ulna and radius at the elbow joint. The elbow is a complex hinge joint between the end of the humerus and the ends of the radius and ulna."  # Paraphrased
is_plagiarized, similarity_score = check_plagiarism(input_text, reference_text, threshold=0.8)

print(f"Input Text: {input_text}")
print(f"Reference Text: {reference_text}")
print(f"Similarity Score: {similarity_score:.4f}")
print(f"Plagiarism Detected: {is_plagiarized}")