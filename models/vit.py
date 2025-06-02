################## ##################
## Based on https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/vit/modeling_vit.py#L627
################## ##################

import torch
import torch.nn as nn
from typing import Optional
import math
from transformers import ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTPreTrainedModel

class ViT(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.default_res = config.image_size
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=False) ### CHANGED FROM ORIGINAL: No mask token

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:

        input_res = x.shape[-1]
        interpolate_pos_encoding = True if input_res != self.default_res else False

        outputs = self.vit(
            x,
            bool_masked_pos=None,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=False
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        return reconstructed_pixel_values