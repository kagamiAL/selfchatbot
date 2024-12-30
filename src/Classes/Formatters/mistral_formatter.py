from Classes.Formatters.formatter import Formatter
from typing import override
from Classes.TypeDicts import MessagePacket


class MistralFormatter(Formatter):
    MODEL_NAME = "MistralForCausalLM"
    MODEL_LABEL = "system"
    USER_LABEL = "user"

    @override
    def format_train(self, packet: MessagePacket) -> str:
        return f"{self.tokenizer.bos_token}{packet['role']}: {packet['content']}{self.tokenizer.eos_token}"
