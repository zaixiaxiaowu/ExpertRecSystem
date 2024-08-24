import streamlit as st
from typing import Optional


def add_chat_message(role: str, message: str, avatar: Optional[str] = None):
    """Add a chat message to the chat history.

    Args:
        `role` (`str`): The role of the message.
        `message` (`str`): The message to be added.
        `avatar` (`Optional[str]`): The avatar of the agent. If `avatar` is `None`, use the default avatar.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})
    if avatar is not None:
        st.chat_message(role, avatar=avatar).markdown(message)
    else:
        st.chat_message(role).markdown(message)


def get_color(agent_type: str) -> str:
    """Get the color of the agent.

    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The color name of the agent.
    """
    if "ProjectAnalyst" in agent_type:
        return "green"
    elif "Recommender" in agent_type:
        return "rainbow"
    elif "Explainer" in agent_type:
        return "red"
    elif "Searcher" in agent_type:
        return "blue"
    else:
        return "orange"


def get_avatar(agent_type: str) -> str:
    """Get the avatar of the agent.

    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The avatar of the agent.
    """
    if "ProjectAnalyst" in agent_type:
        return "👩‍🏫"
    elif "Recommender" in agent_type:
        return "👩‍💼"
    elif "Explainer" in agent_type:
        return "👩‍💻"
    elif "Searcher" in agent_type:
        return "🔍"
    else:
        return "👩‍🏫"


def get_name(agent_type: str) -> str:

    if "ProjectAnalyst" in agent_type:
        return "项目分析器"
    elif "Recommender" in agent_type:
        return "专家排序器"
    elif "Explainer" in agent_type:
        return "推荐解释器"
    elif "Searcher" in agent_type:
        return "专家召回器"
    else:
        return "专家分析器"
