import json
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, Optional, TYPE_CHECKING
from langchain.prompts import PromptTemplate
from ExpertRecSystem.utils import read_prompts
from ExpertRecSystem.llms import BaseLLM, AnyOpenAILLM

if TYPE_CHECKING:
    from ExpertRecSystem.system import System


class Agent(ABC):
    """
    The base class of agents. We use the `forward` function to get the agent output. Use `get_LLM` to get the base large language model for the agent.
    """

    def __init__(
        self,
        prompts: dict = dict(),
        prompt_config: Optional[str] = None,
        web_demo: bool = False,
        system: Optional["System"] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the agent.

        Args:
            `prompts` (`dict`, optional): A dictionary of prompts for the agent. Will be read from the prompt config file if `prompt_config` is not `None`. Defaults to `dict()`.
            `prompt_config` (`Optional[str]`): The path to the prompt config file. Defaults to `None`.
            `web_demo` (`bool`, optional): Whether the agent is used in a web demo. Defaults to `False`.
            `system` (`Optional[System]`): The system that the agent belongs to. Defaults to `None`.
            `dataset` (`Optional[str]`): The dataset that the agent is used on. Defaults to `None`.
        """
        self.json_mode: bool
        self.system = system
        if prompt_config is not None:
            prompts = read_prompts(prompt_config)
        self.prompts = prompts
        self.web_demo = web_demo
        if self.web_demo:
            assert self.system is not None, "System not found."

    def observation(self, message: str, log_head: str = "") -> None:
        """Log the message.

        Args:
            `message` (`str`): The message to log.
            `log_head` (`str`): The log head. Defaults to `''`.
        """
        if self.web_demo:
            self.system.log(log_head + message, agent=self)
        else:
            logger.debug(f"Observation: {message}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the agent.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `Any`: The agent output.
        """
        raise NotImplementedError("Agent.forward() not implemented")

    def get_LLM(
        self, config_path: Optional[str] = None, config: Optional[dict] = None
    ) -> BaseLLM:
        """Get the base large language model for the agent.

        Args:
            `config_path` (`Optional[str]`): The path to the config file of the LLM. If `config` is not `None`, this argument will be ignored. Defaults to `None`.
            `config` (`Optional[dict]`): The config of the LLM. Defaults to `None`.
        Returns:
            `BaseLLM`: The LLM.
        """
        if config is None:
            assert config_path is not None
            with open(config_path, "r") as f:
                config = json.load(f)
        model_type = config["model_type"]
        del config["model_type"]
        if model_type != "api":
            raise NotImplementedError("Agents do not support opensource llm currently")
        else:
            return AnyOpenAILLM(**config)
