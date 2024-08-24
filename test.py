from ExpertRecSystem.system import CollaborationSystem
from ExpertRecSystem.utils import init_openai_api, read_json
import torch

if __name__ == "__main__":
    init_openai_api(read_json("config/openai-api.json"))
    system_config = "config/systems/chat.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = CollaborationSystem(config_path=system_config, device=device)

    user_input = [
        "大连理工大学智能离子膜规模化制造装置采购项目",
        "大连理工大学拟采购智能离子膜规模化制造装置一套。该装置可对智能离子膜制备全过程进行调控及综合评价，实现智能离子膜规模化制造。",
    ]
    system(user_input, top_k=5, num=3)
