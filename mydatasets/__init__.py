# -*- coding:utf-8 -*-
# create: 2021/6/8
from .docparser_dataset import DocParser, DataCollatorForDocParserDataset


def get_dataset(dataset_args):
    dataset_type = dataset_args.get("type")
    dataset = eval(dataset_type)(**dataset_args)
    return dataset

