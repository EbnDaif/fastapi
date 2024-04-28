from typing import List
import numpy as np
import torch
import torch.nn.functional as F


def calculate_cosine_similarity(vector1: List[float], vector2: List[float], threshold: float = 0.8) -> (float, bool):
    # Convert input lists to NumPy arrays
    array1 = np.array(vector1, dtype=np.float32)
    array2 = np.array(vector2, dtype=np.float32)

    # Convert NumPy arrays to PyTorch tensors
    tensor1 = torch.tensor(array1)
    tensor2 = torch.tensor(array2)

    # Reshape the tensors to ensure they have the same shape
    tensor1 = tensor1.view(1, -1)
    tensor2 = tensor2.view(1, -1)

    # Normalize the vectors (optional but recommended for cosine similarity)
    tensor1 = F.normalize(tensor1, p=2, dim=1)
    tensor2 = F.normalize(tensor2, p=2, dim=1)

    # Calculate cosine similarity between tensor1 and tensor2
    similarity_score = torch.mm(tensor1, tensor2.t()).item()

    # Check if similarity_score is greater than or equal to the threshold
    is_match = similarity_score >= 0.96

    return similarity_score, is_match
