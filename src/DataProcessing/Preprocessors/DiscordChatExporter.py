import re
from DataProcessing import process_methods


def preprocess(text: str, params: dict) -> str:
    """
    Preprocess DiscordChatExporter data

    Args:
        text (str): text to preprocess

    Returns:
        str: preprocessed text
    """
    if not params:
        raise FileNotFoundError("data.json not found")
    text = process_methods.remove_all_emoji(text)
    text = process_methods.remove_links(text)
    text = process_methods.remove_braced_content(text)
    text = process_methods.normalize_whitespace(text)
    pre_processed = []
    people_labels = ["User:", "You:"]
    pattern = r"\[\d{4}-\d{2}-\d{2} \d{1,2}:\d{2} [AP]M\]\s+(.*)"
    for line in text.split("\n"):
        match = re.search(pattern, line)
        if match:
            pre_processed.append(
                people_labels[int(match.group(1) == params["username"])]
            )
        else:
            pre_processed.append(line)
    processed = []
    # I don't care that this is O(n^2)
    i = pre_processed.index(people_labels[0])
    current = None
    for j in range(i, len(pre_processed)):
        if pre_processed[j] in people_labels:
            if pre_processed[j] != current:
                if (
                    j + 1 >= len(pre_processed)
                    or pre_processed[j + 1] not in people_labels
                ):
                    current = pre_processed[j]
                    processed.append(current)
        else:
            processed[-1] += " " + pre_processed[j]
    return "\n".join(processed)
