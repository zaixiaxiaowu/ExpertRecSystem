import os
import streamlit as st
import torch
import sys
from loguru import logger
from ExpertRecSystem.utils import init_openai_api, read_json
from ExpertRecSystem.system import *
from ExpertRecSystem.pages.chat import chat_config

# Set up logging with loguru
logger.remove()
logger.add(sys.stderr, level="DEBUG")  # Log to stderr with DEBUG level
os.makedirs("logs", exist_ok=True)  # Ensure the logs directory exists
logger.add(
    "logs/web_{time:YYYY-MM-DD:HH:mm:ss}.log", level="DEBUG"
)  # Log to file with DEBUG level


def demo() -> None:
    """
    Set up and run the Streamlit demo for the expert recommendation system.

    This function initializes the OpenAI API, sets up the device for computation (CPU or GPU),
    configures the Streamlit page layout, and allows the user to select a configuration file and parameters
    for running the expert recommendation system.

    The function sets up the Streamlit sidebar with options to select a configuration file,
    specify the number of experts to recall, and the number of experts to recommend.

    The selected configuration is then passed to the `chat_config` function to run the recommendation system.
    """
    # Initialize OpenAI API with configuration
    init_openai_api(read_json("config/openai-api.json"))

    # Set up the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Computation device set to: {device}")

    # Configure the Streamlit page
    st.set_page_config(
        page_title="大模型驱动的高校采购评审专家推荐系统",
        page_icon="🧠",
        layout="wide",
    )

    # Set up the sidebar in Streamlit
    st.sidebar.title("大模型驱动的高校采购评审专家推荐系统")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Load configuration files from the specified directory
    config_dir = os.path.join("config", "systems")
    config_files = os.listdir(config_dir)

    # Sidebar option to select a configuration file
    st.sidebar.markdown(
        "<h4 style='text-align: left; font-weight: bold;'>选择配置文件</h4>",
        unsafe_allow_html=True,
    )
    config_file = st.sidebar.selectbox("", config_files)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Sidebar slider to select the number of experts to recall
    st.sidebar.markdown(
        "<h4 style='text-align: left; font-weight: bold;'>召回专家数</h4>",
        unsafe_allow_html=True,
    )
    top_k = st.sidebar.slider("", min_value=10, max_value=50, value=10)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Sidebar slider to select the number of experts to recommend
    st.sidebar.markdown(
        "<h4 style='text-align: left; font-weight: bold;'>推荐专家数</h4>",
        unsafe_allow_html=True,
    )
    num = st.sidebar.slider("", min_value=1, max_value=7, value=3)

    # Run the chat configuration with the selected parameters
    chat_config(
        system_type=SYSTEMS[0],
        config_path=os.path.join(config_dir, config_file),
        top_k=top_k,
        num=num,
        device=device,
    )
