import torch
import sys
import os
import pandas as pd
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModel
from ExpertRecSystem.utils import (
    read_expert_data,
    load_faiss_index,
    get_project_embedding,
    find_similar_experts,
    read_json,
)

# Set up logging with loguru
logger.remove()
logger.add(sys.stderr, level="DEBUG")  # Log to stderr with DEBUG level
os.makedirs("logs", exist_ok=True)  # Ensure the logs directory exists
logger.add(
    "logs/recall_{time:YYYY-MM-DD:HH:mm:ss}.log", level="DEBUG"
)  # Log to file with DEBUG level


def evaluate_model(
    test_data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    index,
    expert_data: pd.DataFrame,
    device: torch.device,
):
    """
    Evaluate the recommendation model by comparing the predicted expert rankings against the true expert.

    Args:
        `test_data` (`pd.DataFrame`): DataFrame containing test project descriptions and true expert IDs.
        `tokenizer` (`AutoTokenizer`): Tokenizer for processing the project descriptions.
        `model` (`AutoModel`): Pre-trained model for generating embeddings from project descriptions.
        `index` (`faiss.Index`): FAISS index containing embeddings of experts for similarity search.
        `expert_data` (`pd.DataFrame`): DataFrame containing expert data, including expert IDs.
        `device` (`torch.device`): The device (CPU or GPU) on which the model will run.

    Returns:
        `tuple`: A tuple containing:
            - `hit_rates` (`dict`): A dictionary of hit rates at different cutoff ranks (Top-10, Top-20, etc.).
            - `average_rank` (`float`): The average rank of the true expert in the predicted list.
    """
    hit_counts = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 100: 0}
    ranks = []
    total_count = 0

    # Iterate over each project in the test data
    for _, row in test_data.iterrows():
        project_description = row["project_name"] + " " + row["project_infos"]
        true_expert_id = row["expert_id"]

        # Generate embedding for the project
        project_embedding = get_project_embedding(
            project_description, tokenizer, model, device
        )

        # Find similar experts using FAISS
        cosine_similarities, indices = find_similar_experts(
            project_embedding, index, top_k=1000
        )

        # Retrieve the expert IDs corresponding to the top similar experts
        expert_ids = expert_data.iloc[indices[0]]["expert_id"].values

        # Calculate hit counts for each top-k cutoff
        for k in hit_counts.keys():
            if true_expert_id in expert_ids[:k]:
                hit_counts[k] += 1

        # Determine the rank of the true expert
        rank = np.where(expert_ids == true_expert_id)[0]
        if len(rank) > 0:
            ranks.append(rank[0] + 1)
        else:
            ranks.append(1001)

        total_count += 1

        # Log detailed debug information
        logger.debug(f"Project: {project_description}")
        logger.debug(f"True Expert ID: {true_expert_id}")
        logger.debug(f"True Expert Rank: {rank}")
        logger.debug(f"True Expert Cosine Similarity: {cosine_similarities[rank]}")
        for idx, (expert_id, similarity) in enumerate(
            zip(expert_ids[:5], cosine_similarities[:5])
        ):
            logger.debug(
                f"Rank {idx + 1}: Expert ID: {expert_id}, Cosine Similarity: {similarity:.4f}"
            )
        logger.debug("")

    # Calculate hit rates and average rank
    hit_rates = {k: (hit_counts[k] / total_count) * 100 for k in hit_counts}
    average_rank = np.mean(ranks)

    # Log the evaluation results
    for k, rate in hit_rates.items():
        logger.debug(f"Hit Rate at Top {k}: {rate:.2f}%")

    logger.debug(f"Average Rank: {average_rank:.2f}")
    logger.debug(f"Concrete Rank: {ranks}")

    return hit_rates, average_rank


if __name__ == "__main__":
    # Load configuration and data paths
    config_path = read_json("config/systems/recall.json")
    data_path = config_path["data_path"]
    model_path = config_path["emb_model_path"]
    index_path = config_path["index_path"]
    test_data_path = config_path["test_data_path"]

    # Set up device for model inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model from pre-trained model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    # Load FAISS index and expert data
    index = load_faiss_index(index_path)
    expert_data = read_expert_data(data_path)

    # Load test data
    test_data = pd.read_csv(test_data_path, encoding="utf-8")

    # Evaluate the model
    evaluate_model(test_data, tokenizer, model, index, expert_data, device)

    print(f"Evaluation completed.")
