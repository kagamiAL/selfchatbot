from Classes.Formatters.formatter import Formatter
from typing import override


class GPT2Formatter(Formatter):
    MODEL_NAME = "GPT2LMHeadModel"
    MODEL_LABEL = "Model:"
    USER_LABEL = "User:"

    @override
    def format_prompt(self, text: str) -> str:
        return f"{self.tokenizer.eos_token}{text.strip()}"

    @override
    def format_train(self, text: str) -> str:
        return self.format_for_prompt(text)

    @override
    def format_block(self, block: list[str]) -> str:
        return (
            "".join(self.format_prompt(line) for line in block)
            + self.tokenizer.eos_token
        )
