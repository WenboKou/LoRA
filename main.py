import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother


@dataclass
class DataArguments:
    train_data_path: str = field(default=None)
    eval_data_path: str = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2-0.5B")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=64,
        metadata={
            "help": "Maximum sequence length."
        }
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_module: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj"
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


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


if __name__ == "__main__":
    # parser = transformers.HfArgumentParser(
    #     (
    #         DataArguments,
    #         ModelArguments,
    #         TrainingArguments,
    #         LoraArguments
    #     )
    # )
    #
    # (data_args, model_args, training_args, lora_args) = parser.parse_args_into_dataclasses()
    #
    # print("data_args: ", data_args)
    # print("model_args: ", model_args)
    # print("training_args: ", training_args)
    # print("lora_args: ", lora_args)
    training_args = TrainingArguments(output_dir="output_qwen")
    model_args = ModelArguments()
    lora_args = None

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        ) if training_args.use_lora and lora_args.q_lora
        else None,
        low_cpu_mem_usage=True
    )