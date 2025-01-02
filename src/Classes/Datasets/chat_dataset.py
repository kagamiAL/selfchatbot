from torch.utils.data import Dataset
import torch


def get_longest_token_length(data: list[str], tokenizer) -> int:
    """Returns the longest token length in the dataset

    Args:
        data (list[str]): The dataset
        tokenizer (_type_): The tokenizer

    Returns:
        int: The longest token length
    """
    return max(len(tokenizer.encode(line)) for line in data)


class ChatDataset(Dataset):
    input_ids: list[torch.tensor]
    attn_masks: list[torch.tensor]

    def __init__(self, text: str, tokenizer):
        data = text.split("\n\n")
        self.max_length = get_longest_token_length(data, tokenizer)
        self.data = data
        self.input_ids = []
        self.attn_masks = []
        tokenizer.pad_token = tokenizer.eos_token
        for prompt in data:
            encodings = tokenizer.encode_plus(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,  # Max tokens allowed
                return_tensors="pt",
            )
            self.input_ids.append(encodings["input_ids"])
            self.attn_masks.append(encodings["attention_mask"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.input_ids[index], self.attn_masks[index]
