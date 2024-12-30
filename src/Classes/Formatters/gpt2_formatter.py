from Classes.Formatters.formatter import Formatter
from typing import override
from Classes.TypeDicts import MessagePacket


class GPT2Formatter(Formatter):
    MODEL_NAME = "GPT2LMHeadModel"
    MODEL_LABEL = "Model"
    USER_LABEL = "User"

    @override
    def format_train(self, packet: MessagePacket) -> str:
        return f"{self.tokenizer.eos_token}{packet["role"]}: {packet['content']}"

    @override
    def format_block(self, block: list[MessagePacket]) -> str:
        return super().format_block(block) + self.tokenizer.eos_token
