import os
import sys
import torch
from loguru import logger
from argparse import ArgumentParser
from ExpertRecSystem.system import CollaborationSystem
from ExpertRecSystem.utils import init_openai_api, read_json


def main(user_input, top_k, num):
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/main_{time:YYYY-MM-DD:HH:mm:ss}.log", level="DEBUG")

    init_openai_api(read_json("config/openai-api.json"))
    system_config = "config/systems/chat.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    system = CollaborationSystem(config_path=system_config, device=device)
    logger.debug("Initializing CollaborationSystem")

    logger.debug(f"Received user input: {user_input}")
    logger.debug(f"Top K: {top_k}, Num: {num}")

    logger.debug("Running the CollaborationSystem")
    results = system(user_input, top_k=top_k, num=num)

    logger.debug("System execution completed")
    logger.debug(f"Results: {results}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--user_input", nargs="+", required=True, help="User input for the system"
    )
    parser.add_argument(
        "--top_k", type=int, required=True, help="Number of experts to recall"
    )
    parser.add_argument(
        "--num", type=int, required=True, help="Number of experts to recommend"
    )

    args = parser.parse_args()

    main(args.user_input, args.top_k, args.num)
