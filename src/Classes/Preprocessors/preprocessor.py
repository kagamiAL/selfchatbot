import re
from transformers import AutoTokenizer
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    DATA_FORMAT: str = "Default"

    tokenizer: AutoTokenizer

    COMBINE_TIME: int = 5 * 60
    BLOCK_SPLIT_TIME: int = 60 * 60
    MAXIMUM_LENGTH: int = 1024
    MINIMUM_LENGTH: int = 100

    def __init__(self, params: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(params["model"])

    @abstractmethod
    def normalize(self, text: str) -> list[str]:
        """Normalizes a piece of text before preprocessing

        Args:
            text (str): the text to normalize

        Returns:
            list[str]: the normalized text split into lines
        """
        raise NotImplementedError

    def __split_block_by_token_size(self, block: list[str]) -> list[list[str]]:
        """Splits a block if its token max length is exceeded, also ignores blocks that are < minimum length

        Args:
            block (list[str]): the block

        Returns:
            list[list[str]]: the split block(s)
        """
        encoded = self.tokenizer.encode("\n".join(block) + "\n")
        length = len(encoded)
        if length < self.MINIMUM_LENGTH or len(block) == 1:
            return []
        if length > self.MAXIMUM_LENGTH:
            m = len(block) // 2
            return self.__split_block_by_token_size(
                block[:m]
            ) + self.__split_block_by_token_size(block[m:])
        return [block]

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
        processed_data: list[list[str]] = []
        prev_time, prev_label = 0, None
        # I DO NOT KNOW WHY THIS DOES NOT ERROR, IT JUST WORKS, IM LEAVING IT ALONE
        # THIS IS MESSY I KNOW
        for string in strings:
            epoch_time = int(re.search(r"\d+", string).group())
            label = re.search(r"U|Y", string).group()
            no_epoch = string.lstrip("0123456789 ")
            raw_string = re.search(r"^[YU]:\s*(.*)", no_epoch).group(1)
            time_diff = epoch_time - prev_time
            if time_diff > self.BLOCK_SPLIT_TIME or not processed_data:
                processed_data.append([])
            if prev_label == label and time_diff < self.COMBINE_TIME:
                processed_data[-1][-1] += f" {raw_string.lower()}"
            else:
                processed_data[-1].append(
                    f"{label}: {raw_string[0].upper()}{raw_string[1:]}"
                )
            prev_time = epoch_time
            prev_label = label
        final_processed = []
        for processed in processed_data:
            final_processed += self.__split_block_by_token_size(processed)
        return "\n\n".join(map("\n".join, final_processed))

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
