import os
import streamlit as st
import time
from loguru import logger
from ExpertRecSystem.system import *
from ExpertRecSystem.utils import read_json, add_chat_message


def scan_list(config: list) -> bool:
    """
    Recursively scan a list to check if it contains any JSON file paths and validate them.

    Args:
        `config` (`list`): A list of items, which can be strings, lists, or dictionaries.

    Returns:
        `bool`: Returns True if all JSON files are valid, False otherwise.
    """
    for _, item in enumerate(config):
        if isinstance(item, dict):
            if not scan_dict(item):
                return False
        elif isinstance(item, list):
            if not scan_list(item):
                return False
        elif isinstance(item, str):
            if os.path.isfile(item) and item.endswith(".json"):
                if not check_json(item):
                    return False
    return True


def scan_dict(config: dict) -> bool:
    """
    Recursively scan a dictionary to check if it contains any JSON file paths and validate them.

    Args:
        `config` (`dict`): A dictionary with values that can be strings, lists, or other dictionaries.

    Returns:
        `bool`: Returns True if all JSON files are valid, False otherwise.
    """
    for key in config:
        if isinstance(config[key], str):
            if os.path.isfile(config[key]) and config[key].endswith(".json"):
                if not check_json(config[key]):
                    return False
        elif isinstance(config[key], dict):
            if not scan_dict(config[key]):
                return False
        elif isinstance(config[key], list):
            if not scan_list(config[key]):
                return False
    return True


def check_json(config_path: str) -> bool:
    """
    Check if a JSON file at the given path is valid and meets specific requirements.

    Args:
        `config_path` (`str`): The path to the JSON configuration file.

    Returns:
        `bool`: Returns True if the JSON file is valid, False otherwise.
    """
    config = read_json(config_path)
    if "model_type" in config and config["model_type"] == "opensource":
        assert "model_path" in config, "model_path is required for OpenSource models"
        st.markdown(f'`{config_path}` requires `{config["model_path"]}` models.')
        return False
    if "model_path" in config:
        st.markdown(f'`{config_path}` requires `{config["model_path"]}` models.')
        return False
    return scan_dict(config)


def check_config(config_path: str) -> bool:
    """
    Check if the system configuration is valid and meets the environment's requirements.

    Args:
        `config_path` (`str`): The path to the system configuration file.

    Returns:
        `bool`: Returns True if the configuration is valid, False otherwise.
    """
    import torch

    if torch.cuda.is_available():
        return True
    else:
        return check_json(config_path)


def get_system(system_type: type[System], config_path: str, device: str) -> System:
    """
    Initialize and return a system object based on the provided configuration.

    Args:
        `system_type` (`type[System]`): The class type of the system to be initialized.
        `config_path` (`str`): The path to the configuration file for the system.
        `device` (`str`): The device (e.g., "cpu", "cuda") on which the system will run.

    Returns:
        `System`: An instance of the system initialized with the provided configuration.
    """
    return system_type(config_path=config_path, web_demo=True, device=device)


def chat_config(
    system_type: type[System], config_path: str, top_k: int, num: int, device: str
) -> None:
    """
    Configure the chat interface and manage session state for the system.

    Args:
        `system_type` (`type[System]`): The class type of the system to be initialized.
        `config_path` (`str`): The path to the configuration file for the system.
        `top_k` (`int`): The number of top experts to recall.
        `num` (`int`): The number of experts to recommend.
        `device` (`str`): The device (e.g., "cpu", "cuda") on which the system will run.
    """
    checking = check_config(config_path)
    if not checking:
        st.error(
            f"This config file requires OpenSource models, which are not supported on this machine (without CUDA toolkit)."
        )
        return
    renew = False
    if "config_path" not in st.session_state:
        logger.debug(f"New config path: {config_path}")
        st.session_state.config_path = config_path
        renew = True
    elif st.session_state.config_path != config_path:
        logger.debug(f"Change config path: {config_path}")
        st.session_state.config_path = config_path
        renew = True
    elif "system" not in st.session_state:
        logger.debug(f"New system")
        renew = True
    if renew:
        system = get_system(system_type, config_path, device)
        st.session_state.config_path = config_path
        st.session_state.system = system
    else:
        system = st.session_state.system
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    assert isinstance(st.session_state.chat_history, list)
    chat_page(system=system, top_k=top_k, num=num)


def chat_page(system: CollaborationSystem, top_k: int, num: int) -> None:
    """
    Render the chat interface for interacting with the system.

    Args:
        `system` (`CollaborationSystem`): The system to interact with through the chat interface.
        `top_k` (`int`): The number of top experts to recall.
        `num` (`int`): The number of experts to recommend.
    """
    if "project_name" not in st.session_state:
        st.session_state.project_name = ""
    if "project_description" not in st.session_state:
        st.session_state.project_description = ""

    for chat in st.session_state.chat_history:
        if isinstance(chat["message"], str):
            st.chat_message(chat["role"]).markdown(chat["message"])
        elif isinstance(chat["message"], list):
            with st.chat_message(chat["role"]):
                for message in chat["message"]:
                    st.markdown(f"{message}")
        else:
            st.error("Invalid message format")
            st.write(chat)
            continue

    logger.debug("Initialization complete!")

    with st.form("project_form"):
        project_name = st.text_input(
            "请输入项目名称",
            value="",
            label_visibility="hidden",
            placeholder="请输入项目名称",
        )
        project_description = st.text_area(
            "请输入项目简介",
            value="",
            label_visibility="hidden",
            placeholder="请输入项目简介",
        )
        submit_button = st.form_submit_button("提交")

    if submit_button:
        if project_name and project_description:
            st.session_state.project_name = project_name
            st.session_state.project_description = project_description

            prompt = [project_name, project_description]
            add_chat_message("user", f"项目名称: {prompt[0]}  \n项目简介: {prompt[1]}")

            with st.chat_message("assistant"):
                st.markdown("#### 系统正在运行...")
                response = system(user_input=prompt, top_k=top_k, num=num)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "message": ["#### 系统正在运行..."] + system.web_log,
                    }
                )
            add_chat_message("assistant", response)
            st.session_state.project_name = ""
            st.session_state.project_description = ""
            time.sleep(8)
            st.rerun()
        else:
            st.warning("请填写完整的项目名称和简介！")
