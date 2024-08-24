import json
import streamlit as st
from typing import Any, Optional
from loguru import logger
from ExpertRecSystem.system.base import System
from transformers import AutoTokenizer, AutoModel
from ExpertRecSystem.agents import (
    Agent,
    ProjectAnalyst,
    ExpertAnalyst,
    Explainer,
    Recommender,
)
from ExpertRecSystem.utils import (
    format_chat_history,
    get_project_embedding,
    load_faiss_index,
    find_similar_experts,
    read_expert_data,
    read_json,
)


class CollaborationSystem(System):
    """
    CollaborationSystem is a specialized system for expert recommendation, project analysis, and explanation.
    It integrates multiple agents, including ProjectAnalyst, ExpertAnalyst, Recommender, and Explainer,
    to provide comprehensive functionality for expert recommendation tasks.
    """

    def init(self, *args, **kwargs) -> None:
        """
        Initialize the CollaborationSystem with necessary components such as agents, FAISS index,
        expert data, tokenizer, and model. The initialization process ensures all required components
        are loaded and ready for use.

        Args:
            `*args` (`Any`): Additional positional arguments.
            `**kwargs` (`Any`): Additional keyword arguments, including the device on which the model will run.
        """
        assert "agents" in self.config, "Agents are required."
        self.manager_kwargs = {}
        self.device = kwargs["device"]
        self.init_agents(self.config["agents"])
        self.recall_config = read_json(self.config["recall_config"])
        self.index = load_faiss_index(self.recall_config["index_path"])
        self.expert_data = read_expert_data(self.recall_config["description_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.recall_config["emb_model_path"]
        )
        self.model = AutoModel.from_pretrained(self.recall_config["emb_model_path"]).to(
            self.device
        )

    def init_agents(self, agents: dict[str, dict]) -> None:
        """
        Initialize the agents specified in the configuration.

        Args:
            `agents` (`dict[str, dict]`): A dictionary where the key is the agent name and the value is the agent's configuration.

        Raises:
            `ValueError`: If the agent specified in the configuration is not supported.
        """
        self.agents: dict[str, Agent] = dict()
        for agent, agent_config in agents.items():
            try:
                agent_class = globals()[agent]
                assert issubclass(
                    agent_class, Agent
                ), f"Agent {agent} is not a subclass of Agent."
                self.agents[agent] = agent_class(**agent_config, **self.agent_kwargs)
            except KeyError:
                raise ValueError(f"Agent {agent} is not supported.")

    @property
    def project_analyst(self) -> Optional[ProjectAnalyst]:
        """
        Return the ProjectAnalyst agent if it is available.

        Returns:
            `Optional[ProjectAnalyst]`: The ProjectAnalyst agent instance, or None if it is not available.
        """
        if "ProjectAnalyst" not in self.agents:
            return None
        return self.agents["ProjectAnalyst"]

    @property
    def expert_analyst(self) -> Optional[ExpertAnalyst]:
        """
        Return the ExpertAnalyst agent if it is available.

        Returns:
            `Optional[ExpertAnalyst]`: The ExpertAnalyst agent instance, or None if it is not available.
        """
        if "ExpertAnalyst" not in self.agents:
            return None
        return self.agents["ExpertAnalyst"]

    @property
    def recommender(self) -> Optional[Recommender]:
        """
        Return the Recommender agent if it is available.

        Returns:
            `Optional[Recommender]`: The Recommender agent instance, or None if it is not available.
        """
        if "Recommender" not in self.agents:
            return None
        return self.agents["Recommender"]

    @property
    def explainer(self) -> Optional[Explainer]:
        """
        Return the Explainer agent if it is available.

        Returns:
            `Optional[Explainer]`: The Explainer agent instance, or None if it is not available.
        """
        if "Explainer" not in self.agents:
            return None
        return self.agents["Explainer"]

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        """
        Reset the system state, optionally clearing the chat history.

        Args:
            `clear` (`bool`): If True, clear the chat history. Defaults to False.
            `*args` (`Any`): Additional positional arguments for the reset operation.
            `**kwargs` (`Any`): Additional keyword arguments for the reset operation.
        """
        super().reset(*args, **kwargs)
        if clear:
            self._chat_history = []

    def add_chat_history(self, chat: str, role: str) -> None:
        """
        Add a chat message to the system's chat history.

        Args:
            `chat` (`str`): The chat message.
            `role` (`str`): The role of the sender (e.g., "user", "assistant").
        """
        self._chat_history.append((chat, role))

    @property
    def chat_history(self) -> list[tuple[str, str]]:
        """
        Return the formatted chat history.

        Returns:
            `list[tuple[str, str]]`: A list of tuples representing the chat history, where each tuple contains the message and the role of the sender.
        """
        return format_chat_history(self._chat_history)

    def project_description(self, user_input: list[str]) -> Any:
        """
        Use the ProjectAnalyst agent to analyze the project description provided by the user.

        Args:
            `user_input` (`list[str]`): A list containing the project name and project description.

        Returns:
            `Any`: The result from the ProjectAnalyst agent.
        """
        project_name = user_input[0]
        project_infos = user_input[1]
        return self.project_analyst(
            project_name=project_name, project_infos=project_infos
        )

    def recall(self, user_input: list[str], top_k: int) -> list[dict]:
        """
        Recall similar experts based on the project description.

        Args:
            `user_input` (`list[str]`): A list containing the project name and project description.
            `top_k` (`int`): The number of top similar experts to recall.

        Returns:
            `list[dict]`: A list of dictionaries containing expert information, including similarity scores.
        """
        user_input = user_input[0] + ":" + user_input[1]
        emb = get_project_embedding(user_input, self.tokenizer, self.model, self.device)
        sim, indices = find_similar_experts(emb, self.index, top_k=top_k)
        expert_ids = self.expert_data.iloc[indices[0]]["expert_id"].values
        expert_names = self.expert_data.iloc[indices[0]]["expert_name"].values
        specialist = self.expert_data.iloc[indices[0]]["specialist"].values
        description = self.expert_data.iloc[indices[0]]["description"].values
        expert_info_list = []
        for i in range(len(expert_ids)):
            expert_info = {
                "expert_id": expert_ids[i],
                "expert_name": expert_names[i],
                "specialist": specialist[i],
                "description": description[i],
                "similarity": sim[i],
            }
            expert_info_list.append(expert_info)
        self.log("、".join(expert_names), type="Searcher")
        self.log(expert_info_list, type="ExpertAnalyst")
        return expert_info_list

    def display(self, results: dict[str, Any]) -> list[str]:
        """
        Display the ranked experts and return a formatted list of expert details.

        Args:
            `results` (`dict[str, Any]`): The results containing sorted experts.

        Returns:
            `list[str]`: A list of formatted strings representing the experts and their ranks.

        Raises:
            `NotImplementedError`: If the results do not contain the expected keys ('sorted_experts').
        """
        if "sorted_experts" in results:
            message = []
            for e in results["sorted_experts"]:
                if all(key in e for key in ["rank", "name", "specialist"]):
                    message.append(
                        f"**排名**: {e['rank']}, **姓名**: {e['name']}, **专业**: {e['specialist']}  \n"
                    )
                else:
                    raise NotImplementedError(
                        "There is no 'rank'、'name' or 'specialist' in results"
                    )
            self.log(message, self.recommender)
            return message
        else:
            raise NotImplementedError("There is no 'sorted_experts' in results")

    def add_description(
        self, experts: list[dict], results: dict[str, Any], num: int
    ) -> list[dict]:
        """
        Add descriptions to the top N ranked experts.

        Args:
            `experts` (`list[dict]`): A list of expert information dictionaries.
            `results` (`dict[str, Any]`): The results containing sorted experts.
            `num` (`int`): The number of top experts to update with descriptions.

        Returns:
            `list[dict]`: The top N ranked experts with added descriptions.
        """
        experts_dict = {
            expert["expert_name"]: expert["description"] for expert in experts
        }

        for expert_result in results["sorted_experts"][:num]:
            expert_name = expert_result["name"]

            if expert_name in experts_dict:
                expert_result["description"] = experts_dict[expert_name]
        return results["sorted_experts"][:num]

    def forward(
        self,
        user_input: Optional[list[str]] = None,
        reset: bool = True,
        top_k: int = 10,
        num: int = 3,
        **kwargs,
    ) -> Any:
        """
        Execute the forward pass of the system to generate expert recommendations.

        Args:
            `user_input` (`Optional[list[str]]`): A list containing the project name and project description.
            `reset` (`bool`): Whether to reset the system state before processing. Defaults to True.
            `top_k` (`int`): The number of top similar experts to recall. Defaults to 10.
            `num` (`int`): The number of experts to recommend. Defaults to 3.
            `**kwargs` (`Any`): Additional keyword arguments for the forward pass.

        Returns:
            `Any`: The final recommended experts with explanations.
        """
        self.manager_kwargs["history"] = self.chat_history
        if len(user_input) != 2:
            assert "Project name and project information are both needed."
        if reset:
            self.reset()
        self.add_chat_history(user_input, role="user")
        project = self.project_description(user_input)
        self.log(project, self.project_analyst)

        experts = self.recall(user_input, top_k)

        results = json.loads(self.recommender(project=project, experts=experts))
        final_results = self.display(results)
        update_results = self.add_description(experts, results, num)

        explain = self.explainer(project=project, experts=update_results)
        self.log(explain, self.explainer)
        return final_results[:num]
