# -*- coding:utf-8 -*-
# create: @time: 7/6/23 10:28
from functools import partial

from torch import nn, Tensor

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torchvision.models.convnext import LayerNorm2d, CNBlock


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
            self,
            input_channels: int,
            out_channels: Optional[int],
            num_layers: int,
            stride: Union[Tuple[int, int], int],
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.stride = stride

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=cnf.stride, stride=cnf.stride),
                    )
                )

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == '__main__':
    channel_list = [64, 128, 256]
    num_layer_list = [3, 6, 6]
    stride = [(1, 2), (1, 2), (2, 2)]
    model = ConvNeXt(block_setting=[
        CNBlockConfig(input_channels=channel_list[i_layer],
                      out_channels=channel_list[i_layer] * 2,
                      num_layers=num_layer_list[i_layer],
                      stride=stride[i_layer]
                      )
        for i_layer in range(len(num_layer_list))
    ])
