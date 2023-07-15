# -*- coding:utf-8 -*-
# create: @time: 6/6/23 10:30
"""
Dataloader for Pretraining Task of DocParser

Masked Document Reading Step After the knowledge transfer step, we
pre-train our model on the task of document reading. In this pre-training phase,
the model learns to predict the next textual token while conditioning on the
previous textual tokens and the input image. To encourage joint reasoning, we
mask several 32 × 32 blocks representing approximately fifteen percent of the
input image. In fact, in order to predict the text situated within the masked
regions, the model is obliged to understand its textual context.

"""
import os
import os.path
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Sequence

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel

from base.common_util import load_json

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[1:] = input_ids[:-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class DocParser(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        data_root: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
            self,
            data_root: list,
            donut_model: PreTrainedModel,
            processor,
            max_length: int,
            phase: str = "train",
            ignore_id: int = -100,
            task_start_token: str = "<s>",
            prompt_end_token: str = None,
            sort_json_key: bool = True,
            **kwargs
    ):
        super().__init__()

        self.donut_model = donut_model
        self.processor = processor
        self.max_length = max_length
        self.phase = phase
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        gt_info_list = []
        self.img_path_list = []
        print("processing json to token sequence...")
        for data_dir in data_root:
            for gt_info in load_json(data_dir):
                gt_info_list.extend(gt_info)

        self.dataset_length = len(gt_info_list)
        self.gt_token_sequences = []
        self.special_token_list = []

        for gt_info in tqdm(gt_info_list):
            gt_token_sequence = self.json2token(
                gt_info['extract_info'],
                update_special_tokens_for_json_key=self.phase == "train",
                sort_json_key=self.sort_json_key,
            ) + self.processor.tokenizer.eos_token
            self.gt_token_sequences.append(gt_token_sequence)
            self.img_path_list.append(os.path.join(gt_info['filepath'], gt_info['filename']))

        # add special token
        list_of_tokens = [self.task_start_token, self.prompt_end_token]

        self.add_tokens(list_of_tokens)
        self.donut_model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

        # patch config
        self.height, self.width = self.processor.image_processor.size['height'], self.processor.image_processor.size[
            'width']
        self.num_patches = self.height // 32 * self.width // 32
        self.mask_tensor = torch.zeros(3, 32, 32)

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.special_token_list.extend(list_of_tokens)

    def json2token(self, obj: Any,
                   update_special_tokens_for_json_key: bool = True,
                   sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        list_of_tokens = [fr"<s_{k}>", fr"</s_{k}>"]
                        # add extract token
                        self.add_tokens(list_of_tokens)
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.special_token_list:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        try:
            # pixel_tensor
            sample = Image.open(self.img_path_list[idx]).convert("RGB")
            input_tensor = self.processor(sample, random_padding=self.phase == "train",
                                          do_normalize=False,
                                          return_tensors="pt").pixel_values[0]

            # To encourage joint reasoning, we mask several 32 × 32 blocks
            # representing approximately fifteen percent of the input image.
            input_tensor = self.mask_document_patch(input_tensor)

            # input_ids
            processed_parse = self.gt_token_sequences[idx]
            input_ids = self.processor.tokenizer(
                processed_parse,
                add_special_tokens=False,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
        except:
            random_index = random.randrange(self.__len__())
            return self.__getitem__(random_index)
        # model doesn't need to predict pad token
        return input_tensor, labels, processed_parse

    def mask_document_patch(self, pixel_values):
        patch_width = self.width // 32
        sample_idx_list = random.sample(list(range(self.num_patches)), int(self.num_patches * 0.15))
        for sample_id in sample_idx_list:
            row_id = sample_id // patch_width
            col_id = sample_id % patch_width
            pixel_values[:, row_id * 32: (row_id + 1) * 32, col_id * 32: (col_id + 1) * 32] = self.mask_tensor
        return self.processor(pixel_values, return_tensors="pt").pixel_values[0]


@dataclass
class DataCollatorForDocParserDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        # pixel_values
        images = [instance[0] for instance in instances]
        batch['pixel_values'] = torch.stack(images)
        # labels
        labels = [instance[1] for instance in instances]
        batch['labels'] = torch.stack(labels)
        # processed_parse
        batch['processed_parse'] = [instance[2] for instance in instances]
        return batch




