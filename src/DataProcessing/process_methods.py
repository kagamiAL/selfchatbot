import re
import emoji


def normalize_whitespace(text: str) -> str:
    """Remove multiple spaces and newlines

    Args:
        text (str): the text

    Returns:
        str: the normalized text
    """
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def remove_all_emoji(text: str) -> str:
    """Remove all emoji from the text

    Args:
        text (str): the text

    Returns:
        str: the text without emoji
    """
    return emoji.replace_emoji(text)


def remove_links(text: str) -> str:
    """Remove all links

    Args:
        text (str): the text

    Returns:
        str: the text without links
    """
    pattern = r"https?://\S+|www\.\S+"
    return re.sub(pattern, "", text)


def remove_braced_content(text) -> str:
    """Remove all content inside {}

    Args:
        text (str): the text

    Returns:
        str: the text without {}
    """

    pattern = r"\{.*?\}"
    return re.sub(pattern, "", text)
