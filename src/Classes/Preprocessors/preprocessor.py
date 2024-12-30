import re
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from Classes.Formatters.formatter import Formatter, get_formatter
from Classes.TypeDicts import MessagePacket


class Preprocessor(ABC):
    DATA_FORMAT: str = "Default"

    tokenizer: AutoTokenizer
    formatter: Formatter

    COMBINE_TIME: int = 5 * 60
    BLOCK_SPLIT_TIME: int = 60 * 60
    MAXIMUM_LENGTH: int = 1024
    MINIMUM_LENGTH: int = 75

    def __init__(self, params: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(params["model"])
        self.formatter = get_formatter(params["model"], self.tokenizer)

    @abstractmethod
    def normalize(self, text: str) -> list[MessagePacket]:
        """Normalizes a piece of text into a list of MessagePackets

        Args:
            text (str): the text to normalize

        Raises:
            NotImplementedError

        Returns:
            list[MessagePacket]: the normalized text as a list of MessagePackets
        """
        raise NotImplementedError

    def __split_block_by_token_size(self, block: list[str]) -> list[list[str]]:
        """Splits a block if its token max length is exceeded, also ignores blocks that are < minimum length

        Args:
            block (list[str]): the block

        Returns:
            list[list[str]]: the split block(s)
        """
        encoded = self.tokenizer.encode(self.formatter.format_block(block))
        length = len(encoded)
        if length < self.MINIMUM_LENGTH or len(block) == 1:
            return []
        if length > self.MAXIMUM_LENGTH:
            m = len(block) // 2
            return self.__split_block_by_token_size(
                block[:m]
            ) + self.__split_block_by_token_size(block[m:])
        return [block]

    def preprocess_normalized(self, packets: list[MessagePacket]) -> str:
        """Preprocesses a list of packets in the format:
        {
            "time": int,
            "role": str,
            "content": str
        }

        Args:
            packets (list[Packet]): the packets to preprocess

        Returns:
            str: the preprocessed text
        """
        processed_data: list[list[MessagePacket]] = []
        prev_time, prev_label = 0, None
        # I DO NOT KNOW WHY THIS DOES NOT ERROR, IT JUST WORKS, IM LEAVING IT ALONE
        # THIS IS MESSY I KNOW
        for packet in packets:
            epoch_time = packet["time"]
            label = packet["role"]
            raw_string = packet["content"]
            time_diff = epoch_time - prev_time
            if time_diff > self.BLOCK_SPLIT_TIME or not processed_data:
                processed_data.append([])
            if prev_label == label and time_diff < self.COMBINE_TIME:
                processed_data[-1][-1]["content"] += f"\n{raw_string}"
            else:
                processed_data[-1].append(packet)
            prev_time = epoch_time
            prev_label = label
        final_processed = []
        for processed in processed_data:
            final_processed += self.__split_block_by_token_size(processed)
        return "\n\n".join(
            self.formatter.format_block(block) for block in final_processed
        )

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
