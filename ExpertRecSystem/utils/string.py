def format_step(step: str) -> str:
    """Format a step prompt. Remove leading and trailing whitespaces and newlines, and replace newlines with spaces.

    Args:
        `step` (`str`): A step prompt in string format.
    Returns:
        `str`: The formatted step prompt.
    """
    return step.strip("\n").strip().replace("\n", "")


def format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format chat history prompt. Add a newline between each turn in `history`.

    Args:
        `history` (`list[tuple[str, str]]`): A list of turns in the chat history. Each turn is a tuple with the first element being the chat record and the second element being the role.
    Returns:
        `str`: The formatted chat history prompt. If `history` is empty, return `'No chat history.\\n'`.
    """
    if history == []:
        return "No chat history.\n"
    else:
        return (
            "\n"
            + "\n".join([f"{role.capitalize()}: {chat}" for chat, role in history])
            + "\n"
        )
