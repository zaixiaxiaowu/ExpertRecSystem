import torch
import faiss
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def read_expert(data_path: str) -> pd.Series:
    """
    Read expert data from a CSV file, clean it, and generate a text description for each expert.

    Args:
        `data_path` (`str`): The path to the CSV file containing expert data.

    Returns:
        `pd.Series`: A pandas Series containing the generated text descriptions for each expert.
    """
    data = pd.read_csv(data_path, encoding="utf-8")
    data = data.fillna("")  # Replace NaN values with empty strings
    data["expert_text"] = data.apply(
        build_expert_text, axis=1
    )  # Generate expert descriptions
    return data["expert_text"]


def get_embedding(
    texts: pd.Series, tokenizer: AutoTokenizer, model: AutoModel
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using a pre-trained transformer model.

    Args:
        `texts` (`pd.Series`): A pandas Series containing text descriptions of experts.
        `tokenizer` (`AutoTokenizer`): A tokenizer from the transformers library for processing text.
        `model` (`AutoModel`): A pre-trained transformer model from the transformers library.

    Returns:
        `np.ndarray`: A numpy array containing the generated embeddings.
    """
    embeddings = []
    model.to(device)  # Move model to the specified device (GPU or CPU)
    for text in tqdm(texts, desc="Generating embeddings"):
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
        embedding = (
            outputs.last_hidden_state[:, 0, :].cpu().numpy()
        )  # Extract the embedding
        embedding = embedding / np.linalg.norm(
            embedding, axis=1, keepdims=True
        )  # Normalize the embedding
        embeddings.append(embedding)

    return np.vstack(embeddings)  # Combine all embeddings into a single numpy array


def build_expert_text(row: pd.Series) -> str:
    """
    Build a textual description for an expert based on their specialty, workplace, and project history.

    Args:
        `row` (`pd.Series`): A row of data containing information about the expert.

    Returns:
        `str`: A string containing a formatted description of the expert.
    """
    specialty = row["specialist"]
    workplace = row["workplace"]
    projects = row["history_item_name"].split("\n")
    project_infos = row["history_item_info"].split("\n")

    combined_projects = []
    for name, info in zip(projects, project_infos):
        combined_projects.append(f"{name}: {info}")

    expert_text = f"专业:{specialty} 单位:{workplace} " + "项目:".join(
        combined_projects
    )
    return expert_text


def save_to_faiss(embeddings: np.ndarray, index_path: str) -> None:
    """
    Save the generated embeddings to a FAISS index file for efficient similarity search.

    Args:
        `embeddings` (`np.ndarray`): A numpy array containing the generated embeddings.
        `index_path` (`str`): The path where the FAISS index will be saved.
    """
    os.makedirs(
        os.path.dirname(index_path), exist_ok=True
    )  # Ensure the directory exists
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(
        dimension
    )  # Create a FAISS index for cosine similarity search
    index.add(embeddings)  # Add embeddings to the index
    faiss.write_index(index, index_path)  # Save the index to a file


if __name__ == "__main__":
    # Define paths and device
    data_path = "data/raw/all_data.csv"
    model_path = "config/bge-m3"
    index_path = "data/processed/faiss_index_all"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model from pre-trained model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Read expert data and generate embeddings
    data = read_expert(data_path)
    embeddings = get_embedding(data, tokenizer, model)

    # Save embeddings to FAISS index
    save_to_faiss(embeddings, index_path)
    print(f"FAISS index saved to {index_path}")
