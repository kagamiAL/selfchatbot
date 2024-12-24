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


def remove_phone_numbers(text):
    # Regular expression to match phone numbers (e.g., formats like 123-456-7890, (123) 456-7890, 1234567890)
    phone_number_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    cleaned_text = re.sub(phone_number_pattern, "1234567890", text)
    return cleaned_text


def remove_social_media_handles(text):
    # Regular expression to match social media handles (e.g., formats like @username, @username123, @username123_)
    social_media_pattern = r"[@#][\w.]+"
    cleaned_text = re.sub(social_media_pattern, "@anonymous", text)
    return cleaned_text
