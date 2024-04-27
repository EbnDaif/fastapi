from typing import List
import torch
import torch.nn.functional as F

def calculate_cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    # Convert input lists to PyTorch tensors
    tensor1 = torch.tensor(vector1)
    tensor2 = torch.tensor(vector2)
    
    # Reshape the tensors to ensure they have the same shape
    tensor1 = tensor1.view(1, -1)
    tensor2 = tensor2.view(1, -1)

    # Normalize the vectors (optional but recommended for cosine similarity)
    tensor1 = F.normalize(tensor1, p=2, dim=1)
    tensor2 = F.normalize(tensor2, p=2, dim=1)

    # Calculate cosine similarity between tensor1 and tensor2
    similarity_score = torch.mm(tensor1, tensor2.t()).item()  # matrix multiplication

    # Check if similarity_score is less than 1
    if similarity_score > 0.8:
        result = "Same persons"
    else:
        result = "Not matched"

    return similarity_score, result
