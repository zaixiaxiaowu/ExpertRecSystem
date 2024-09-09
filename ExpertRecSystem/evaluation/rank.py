import torch
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from ExpertRecSystem.system import CollaborationSystem
from ExpertRecSystem.utils import init_openai_api, read_json

current_time = datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
log_filename = f"logs/sort_{current_time}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)

if __name__ == "__main__":
    init_openai_api(read_json("config/openai-api.json"))
    system_config = "config/systems/chat.json"
    data_path = "data/raw/test_data.csv"
    output_path = "data/processed/sorted_results.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = CollaborationSystem(config_path=system_config, device=device)
    data = pd.read_csv(data_path)
    results = []

    for _, row in tqdm(
        data.iterrows(), total=data.shape[0], desc="Processing Projects"
    ):
        try:
            project_name = row["project_name"]
            project_info = row["project_infos"]
            project_id = row["project_id"]
            user_input = [project_name, project_info]

            output = system(user_input, top_k=25, num=10)

            results.append(
                {
                    "project_id": project_id,
                    "project_name": project_name,
                    "system_output": output,
                }
            )

        except Exception as e:

            logging.error(f"Error processing {project_name}: {e}")
            continue

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Results saved to {output_path}")
    print(f"Logs saved to {log_filename}")
