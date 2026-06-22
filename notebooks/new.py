import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences to compare
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Quantum mechanics describes subatomic particles.",
    "I love programming in Python.",
    "Python snakes are common in Asia.",
]

# Embed -> shape (5, 384)
embeddings = model.encode(sentences)

# Normalize each row to unit length
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / norms

# Cosine similarity matrix: (5, 384) @ (384, 5) -> (5, 5)
similarity_matrix = normalized @ normalized.T

# Print sentences with labels
labels = [f"S{i+1}" for i in range(len(sentences))]
print("Sentences:")
for label, s in zip(labels, sentences):
    print(f"  {label}: {s}")

# Print the matrix
np.set_printoptions(precision=3, suppress=True)
print("\nCosine similarity matrix:")
print("       " + "   ".join(labels))
for i, row in enumerate(similarity_matrix):
    print(f"{labels[i]}   " + "  ".join(f"{v:6.3f}" for v in row))