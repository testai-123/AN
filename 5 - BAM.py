import numpy as np

def train_BAM(pattern1, pattern2):
    # Transpose patterns for correct matrix multiplication
    pattern1_transpose = pattern1.T
    pattern2_transpose = pattern2.T
    
    # Compute weights
    weights = np.dot(pattern1_transpose, pattern2)
    
    return weights

def recall_forward(input_pattern, weights):

    # Compute recalled pattern
    recalled_pattern = np.sign(np.dot(input_pattern, weights))
    
    return recalled_pattern

def recall_backward(input_pattern, weights):

    # Compute recalled pattern
    recalled_pattern = np.sign(np.dot(input_pattern, weights.T))
    
    return recalled_pattern

# Example patterns
pattern1 = np.array([[1, 1, 1, -1, -1]])  # Pattern A
pattern2 = np.array([[-1, -1, -1, 1, 1]])  # Pattern B

# Train BAM model
weights = train_BAM(pattern1, pattern2)

# Recall forward
output_pattern1 = recall_forward(pattern1, weights)
print(f"Recalled Pattern 1 (Forward): {output_pattern1}")

# Recall backward
output_pattern2 = recall_backward(pattern2, weights)
print(f"Recalled Pattern 2 (Backward): {output_pattern2}")
