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
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    @override
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = self.CHAT_TEMPLATE
        else:
            warn(
                "There is already a chat template in the tokenizer, the roles may be different, please check."
            )

    @override
    def format_block(
        self, block: list[MessagePacket], add_generation_prompt: bool = False
    ) -> str:
        return self.tokenizer.apply_chat_template(
            block, tokenize=False, add_generation_prompt=add_generation_prompt
        )
