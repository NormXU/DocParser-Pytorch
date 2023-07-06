# -*- coding:utf-8 -*-
# create: @time: 7/6/23 16:38
import torch
from transformers import VisionEncoderDecoderConfig, AutoConfig, VisionEncoderDecoderModel, AutoModel
from models import DocParserModel, DocParserConfig

if __name__ == '__main__':
    AutoConfig.register("docparser-swin", DocParserConfig)
    AutoModel.register(DocParserConfig, DocParserModel)

    config = VisionEncoderDecoderConfig.from_pretrained("../models/")
    model = VisionEncoderDecoderModel(config=config)

    # test forward with dummy input
    input_tensor = torch.ones(1, 3, 2560, 1920)
    output = model(input_tensor)
