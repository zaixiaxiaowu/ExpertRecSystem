import streamlit as st
from loguru import logger
from abc import ABC, abstractmethod
from typing import Any, Optional
from ExpertRecSystem.agents import Agent
from ExpertRecSystem.utils import read_json, get_avatar, get_color, get_name


class System(ABC):
    """
    The base class of systems. We use the `forward` function to get the system output. Use `set_data` to set the input, context and ground truth answer. Use `is_finished` to check whether the system has finished. Use `is_correct` to check whether the system output is correct. Use `finish` to finish the system and set the system output.
    """

    def __init__(
        self,
        config_path: str,
        web_demo: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the system.

        Args:
            `config_path` (`str`): The path to the config file of the system.
            `leak` (`bool`, optional): Whether to leak the ground truth answer to the system during inference. Defaults to `False`.
            `web_demo` (`bool`, optional): Whether to run the system in web demo mode. Defaults to `False`.
            `dataset` (`str`, optional): The dataset to run in the system. Defaults to `None`.
        """
        self.config = read_json(config_path)
        self.agent_kwargs = {
            "system": self,
        }
        self.web_demo = web_demo
        self.agent_kwargs["web_demo"] = web_demo
        self.kwargs = kwargs
        self.init(*args, **kwargs)
        self.reset(clear=True)

    @abstractmethod
    def init(self, *args, **kwargs) -> None:
        """Initialize the system.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        """
        raise NotImplementedError("System.init() not implemented")

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the system.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `Any`: The system output.
        """
        raise NotImplementedError("System.forward() not implemented")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.clear_web_log()
        return self.forward(*args, **kwargs)

    def clear_web_log(self) -> None:
        self.web_log = []

    def init_answer(self) -> Any:
        return ""

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        self.scratchpad: str = ""
        self.finished: bool = False
        self.answer = self.init_answer()
        if self.web_demo and clear:
            self.clear_web_log()

    def is_finished(self) -> bool:
        return self.finished

    def finish(self, answer: Any) -> str:
        self.answer = answer
        observation = f"综上所述: {self.answer}"
        self.finish = True
        return observation

    def log(
        self,
        message: str | list,
        agent: Optional[Agent] = None,
        type: str = None,
        logging: bool = True,
    ) -> None:
        """Log the message.

        Args:
            `message` (`str`): The message to log.
            `agent` (`Agent`, optional): The agent to log the message. Defaults to `None`.
            `logging` (`bool`, optional): Whether to use the `logger` to log the message. Defaults to `True`.
        """
        if logging:
            logger.debug(message)
        if self.web_demo:
            if agent is None:
                role = type
            else:
                role = agent.__class__.__name__
            if isinstance(message, str):
                final_message = f"{get_avatar(role)}:{get_color(role)}[**{get_name(role)}**]  \n\n{message}"
                st.markdown(f"{final_message}")
                self.web_log.append(final_message)

            elif isinstance(message, list):
                header_message = (
                    f"{get_avatar(role)}:{get_color(role)}[**{get_name(role)}**]"
                )
                st.markdown(header_message)
                self.web_log.append(header_message)
                if role == "ExpertAnalyst":
                    for item in message:
                        item_mesagge = f"**编号**: {item['expert_id']} **姓名**: {item['expert_name']} **专业**: {item['specialist']}。 {item['description']}  \n"
                        self.web_log.append(item_mesagge)
                        st.markdown(item_mesagge)
                elif role == "Recommender":
                    for item_mesagge in message:
                        self.web_log.append(item_mesagge)
                        st.markdown(item_mesagge)
