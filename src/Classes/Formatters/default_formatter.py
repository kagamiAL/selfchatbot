from Classes.Formatters.formatter import Formatter
from Classes.TypeDicts import MessagePacket
from typing import override
from warnings import warn


class DefaultFormatter(Formatter):
    MODEL_NAME = "Default"
    MODEL_LABEL = "assistant"
    USER_LABEL = "user"
    CHAT_TEMPLATE = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + message['content'] + '<|eot_id|>'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>' }}{% endif %}"
    )

    @override
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        if self.tokenizer.chat_template:
            warn(
                "There already is a chat template set in the tokenizer, this will be overwritten"
            )
        self.tokenizer.chat_template = self.CHAT_TEMPLATE

    @override
    def format_block(self, block: list[MessagePacket]) -> str:
        return self.tokenizer.apply_chat_template(block, tokenize=False)
