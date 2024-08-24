import pandas as pd
from tqdm import tqdm
from ExpertRecSystem.utils import (
    read_json,
    read_expert_data,
    init_openai_api,
    read_prompts,
)
from ExpertRecSystem.agents.expert_analyst import ExpertAnalyst


if __name__ == "__main__":

    init_openai_api(read_json("config/openai-api.json"))
    prompt = read_prompts("config/prompts/agent_prompt/expert_analyst.json")
    model_config_path = "config/agents/expert_analyst.json"
    expert_data = read_expert_data("data/raw/train_data.csv")
    results = []
    expert_analyst = ExpertAnalyst(model_config_path, prompts=prompt)
    for _, row in tqdm(
        expert_data.iterrows(), total=len(expert_data), desc="Processing experts"
    ):
        expert_id = row["expert_id"]
        expert_name = row["expert_name"]
        specialty = row["specialist"]
        projects = row["history_item_name"].split("\n")
        project_infos = row["history_item_info"].split("\n")

        analysis = expert_analyst(
            expert_name=expert_name,
            specialty=specialty,
            projects=projects,
            project_infos=project_infos,
        )
        results.append(
            {
                "expert_id": expert_id,
                "expert_name": expert_name,
                "specialist": specialty,
                "description": analysis,
            }
        )
    df = pd.DataFrame(results)
    file_path = "data/processed/expert_analysis.csv"
    df.to_csv(file_path, index=False)
    print(f"expert analysis saved to {file_path}")
