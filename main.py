import json
from typing import Dict

import torch
import transformers
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother


def preprocess(
        messages,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for msg in messages:
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                truncation=True,
                padding=True,
                max_length=max_len,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = LabelSmoother.ignore_index
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super().__init__()

        messages = [sample["messages"] for sample in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            target_ids=self.target_ids[i],
            attention_mask=self.attention_mask[i]
        )


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: str,
        max_len: int
) -> Dict:
    """make dataset and collator for supervised fine-tuning."""

    train_data = []
    with open(data_args.train_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            train_data.append(json.loads(line))

    eval_data = []
    if data_args.eval_data_path:
        with open(data_args.eval_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                eval_data.append(json.loads(line))

    train_dataset = SupervisedDataset(train_data, tokenizer, max_len)
    eval_dataset = SupervisedDataset(eval_data, tokenizer, max_len) if eval_data else None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
