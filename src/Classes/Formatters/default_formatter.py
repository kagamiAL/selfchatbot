import torch
import torch.nn.functional as F
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

    @override
    def get_prompt_encoding(
        self, block: list[MessagePacket]
    ) -> dict[str, torch.Tensor]:
        return self.tokenizer.apply_chat_template(
            block,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    @override
    def filter_output(self, prompt: torch.Tensor, raw_output: torch.Tensor) -> str:
        max_length = max(len(prompt), len(raw_output))
        padded_prompt = F.pad(prompt, (0, max_length - len(prompt)), value=-1)
        padded_output = F.pad(raw_output, (0, max_length - len(raw_output)), value=-1)
        # Find differing elements while ignoring padding
        difference_mask = padded_prompt != padded_output
        # Get the differing elements
        filtered_encoding = padded_output[difference_mask & (padded_output != -1)]
        return self.tokenizer.decode(filtered_encoding, skip_special_tokens=True)
