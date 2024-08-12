from typing import Dict

import torch
import transformers
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
