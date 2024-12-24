import re
from DataProcessing import process_methods


class Preprocessor:
    DATA_FORMAT: str = "Default"
    COMBINE_TIME: int = 5 * 60
    BLOCK_SPLIT_TIME: int = 60 * 60
    model: str
    max_length: int = 0

    def __init__(self, params: dict):
        self.model = params["model"]

    def normalize(self, text: str) -> list[str]:
        """Normalizes a piece of text before preprocessing

        Args:
            text (str): the text to normalize

        Returns:
            list[str]: the normalized text split into lines
        """
        text = process_methods.remove_all_emoji(text)
        text = process_methods.remove_links(text)
        text = process_methods.remove_braced_content(text)
        text = process_methods.remove_phone_numbers(text)
        text = process_methods.remove_social_media_handles(text)
        text = process_methods.normalize_whitespace(text)
        raise text.splitlines()

    def preprocess_normalized(self, strings: list[str]) -> str:
        """Preprocesses a list of normalized strings in the format:
        Epoch-time(in seconds and int) U: message
        Epoch-time(in seconds and int) Y: message
        Where U and Y are in arbitrary order

        Args:
            strings (list[str]): the list of normalized strings

        Returns:
            str: the preprocessed text
        """
        processed_data = []
        prev_time, prev_label = 0, None
        for string in strings:
            epoch_time = int(re.search(r"\d+", string).group())
            label = re.search(r"U|Y", string).group()
            time_diff = epoch_time - prev_time
            if time_diff > self.BLOCK_SPLIT_TIME:
                processed_data.append([])
            prev_time = epoch_time
        return "\n".join(strings)

    def preprocess(self, text: str) -> str:
        """Preprocesses a piece of text

        Args:
            text (str): the text to preprocess

        Raises:
            NotImplementedError: _description_

        Returns:
            str: the preprocessed text
        """
        return self.preprocess_normalized(self.normalize(text))
