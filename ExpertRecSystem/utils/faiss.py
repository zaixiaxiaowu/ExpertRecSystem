import faiss
import torch
import numpy as np


def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Load a FAISS index from the specified file path.

    Args:
        `index_path` (`str`): The path to the FAISS index file.

    Returns:
        `faiss.Index`: The loaded FAISS index.
    """
    index = faiss.read_index(index_path)
    return index


def get_project_embedding(
    project_description: str, tokenizer, model, device: torch.device
) -> np.ndarray:
    """
    Generate an embedding for the given project description using a pre-trained transformer model.

    Args:
        `project_description` (`str`): The textual description of the project.
        `tokenizer`: The tokenizer to convert the text into token IDs.
        `model`: The pre-trained transformer model to generate the embedding.
        `device` (`torch.device`): The device (CPU or GPU) on which the model will run.

    Returns:
        `np.ndarray`: The normalized embedding of the project description.
    """
    inputs = tokenizer(
        project_description,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding


def find_similar_experts(
    project_embedding: np.ndarray, index: faiss.Index, top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the most similar experts to the project based on the project embedding.

    Args:
        `project_embedding` (`np.ndarray`): The embedding of the project description.
        `index` (`faiss.Index`): The FAISS index containing expert embeddings.
        `top_k` (`int`, optional): The number of top similar experts to retrieve. Defaults to 5.

    Returns:
        `tuple[np.ndarray, np.ndarray]`: A tuple containing:
            - `cosine_similarities` (`np.ndarray`): The cosine similarity scores of the top similar experts.
            - `indices` (`np.ndarray`): The indices of the top similar experts in the FAISS index.
    """
    distances, indices = index.search(project_embedding, top_k)
    cosine_similarities = distances[0]
    return cosine_similarities, indices
