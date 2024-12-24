from torch.utils.data import Dataset
import torch


class ChatDataset(Dataset):
    input_ids: list[torch.tensor]
    attn_masks: list[torch.tensor]

    def __init__(self, text: str, tokenizer):
        split = text.splitlines()
        data = list(map("\n".join, zip(split[0::2], split[1::2])))
        max_length = len(max(data, key=len))
        self.data = data
        self.input_ids = []
        self.attn_masks = []
        tokenizer.pad_token = tokenizer.eos_token
        for prompt in data:
            encodings = tokenizer.encode_plus(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids.append(encodings["input_ids"])
            self.attn_masks.append(encodings["attention_mask"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.input_ids[index], self.attn_masks[index]
