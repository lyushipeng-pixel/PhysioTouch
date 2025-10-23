import sys

import copy

import torch
import numpy as np
from einops import rearrange
from typing import Optional, Tuple, Union
import time

from torch import nn
# from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from transformers import AutoConfig
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING,CLIPVisionModel, CLIPPreTrainedModel, CLIPOutput,CLIPTextModel
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward
from model.process_clip import get_global_value, set_global_value
from model.util.pos_embed import get_2d_sincos_pos_embed
from .mae_model import TactileMAE, TactileVideoMAE

# def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
#     return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
#     caption_loss = contrastive_loss(similarity)
#     image_loss = contrastive_loss(similarity.t())
#     return (caption_loss + image_loss) / 2.0

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer", "CLIPVisionEmbeddings"]

    def __init__(self, args, config, decoder_config, num_frames, add_time_attn, tube_size):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        self.alpha_vl = args.alpha_vl
        self.alpha_vt = args.alpha_vt
        self.alpha_lt = args.alpha_lt
        self.do_mae = True
        self.no_text = args.no_text
        self.cross_alpha = args.cross_alpha
        if args.no_mae == True:
            self.do_mae = False

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        text_model = CLIPTextModel._from_config(text_config, attn_implementation=config._attn_implementation)
        self.text_model = text_model.text_model

        vision_model = CLIPVisionModel._from_config(vision_config, attn_implementation=config._attn_implementation)
        self.vision_model = vision_model.vision_model

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.cross_sensor_head = nn.Linear(self.projection_dim, 1)
        self.bceloss = nn.BCEWithLogitsLoss()
        #self.post_layernorm_touch = nn.LayerNorm(embed_dim, eps=vision_config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.touch_mae_model = TactileVideoMAE(args, config, decoder_config, num_frames, add_time_attn, tube_size)

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


    def clip_loss(self, logits_per_touch_image, logits_per_text_image, logits_per_touch_text):
        touch_image_loss = torch.zeros(1, device = self.device)
        image_touch_loss = torch.zeros(1, device = self.device)
        text_image_loss = torch.zeros(1, device = self.device)
        image_text_loss = torch.zeros(1, device = self.device)
        touch_text_loss = torch.zeros(1, device = self.device)
        text_touch_loss = torch.zeros(1, device = self.device)

        if logits_per_touch_image is not None:
            touch_image_loss = self.contrastive_loss(logits_per_touch_image)
            image_touch_loss = self.contrastive_loss(logits_per_touch_image.t())
        
        if logits_per_text_image is not None:
            text_image_loss = self.contrastive_loss(logits_per_text_image)
            image_text_loss = self.contrastive_loss(logits_per_text_image.t())
        
        if logits_per_touch_text is not None:
            touch_text_loss = self.contrastive_loss(logits_per_touch_text)
            text_touch_loss = self.contrastive_loss(logits_per_touch_text.t())

        if (logits_per_touch_image is not None) or (logits_per_text_image is not None) or (logits_per_touch_text is not None):
            return self.alpha_vl*(text_image_loss + image_text_loss) / 2.0 + self.alpha_vt*(touch_image_loss + image_touch_loss) / 2.0 + self.alpha_lt*(touch_text_loss + text_touch_loss)
        else:
            return None

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        return_dict = self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def get_touch_features(self, x, sensor_type=None, data_type = None):
       
        if data_type == 0:
            touch_features, _, _ = self.touch_mae_model.forward_encoder(x, sensor_type=sensor_type, data_type = data_type, use_mask = False)
        else:
            touch_features, _, _ = self.touch_mae_model.forward_encoder(x[:, :3], sensor_type=sensor_type, data_type = data_type, use_mask = False)
        # print(touch_features.shape)
        # exit(0)

        return touch_features

    def forward(
        self, input_ids=None, attention_mask=None, pixel_values=None, touch_input=None, 
        sensor_type=None, data_type = None, target_sensor_type = None, vision_flag = None, text_flag = None, 
        positive_sample = None, negative_sample = None, pos_sensors=None, neg_sensors=None):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.

        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        return_dict = self.config.use_return_dict

        if positive_sample is not None:
            touch_embeds = self.get_touch_features(touch_input, sensor_type=sensor_type, data_type = data_type)
            touch_embeds = touch_embeds / _get_vector_norm(touch_embeds)

            positive_touch_embeds = self.get_touch_features(positive_sample, sensor_type=pos_sensors, data_type = data_type)
            positive_touch_embeds = positive_touch_embeds / _get_vector_norm(positive_touch_embeds)

            negative_touch_embeds = self.get_touch_features(negative_sample, sensor_type=neg_sensors, data_type = data_type)
            negative_touch_embeds = negative_touch_embeds / _get_vector_norm(negative_touch_embeds)

            positive_vec = torch.mul(touch_embeds, positive_touch_embeds)
            negative_vec = torch.mul(touch_embeds, negative_touch_embeds)

            positive_sim = self.cross_sensor_head(positive_vec).squeeze(1)
            negative_sim = self.cross_sensor_head(negative_vec).squeeze(1)

            positive_label = torch.ones_like(positive_sim).float()
            negative_label = torch.zeros_like(negative_sim).float()

            matching_loss = self.cross_alpha * (self.bceloss(positive_sim, positive_label) + self.bceloss(negative_sim, negative_label))

            return matching_loss



        text_embeds = None
        image_embeds = None
        touch_embeds = None

        vision_indices = None
        text_indices = None
        vision_indices_t = None
        text_indices_v = None

        if vision_flag is not None:
            if vision_flag.sum() > 0:
                vision_indices = torch.nonzero(vision_flag == 1).flatten()
                pixel_values = pixel_values[vision_indices]
            else:
                pixel_values = None
        
        if text_flag is not None:
            if text_flag.sum() > 0:
                text_indices = torch.nonzero(text_flag == 1).flatten()
                input_ids = input_ids[text_indices]
                attention_mask = attention_mask[text_indices]
            else:
                input_ids = None
                attention_mask = None

        if (vision_indices is not None) and (text_indices is not None):
            vision_flag_t = vision_flag[text_indices]
            text_flag_v = text_flag[vision_indices]

            text_indices_v = torch.nonzero(vision_flag_t == 1).flatten()
            vision_indices_t = torch.nonzero(text_flag_v == 1).flatten()
        
        if pixel_values is not None:
            if len(pixel_values.shape) == 1 or pixel_values.shape[-1] == 1:
                pixel_values = None

        if input_ids is not None:
            if len(input_ids.shape) == 1 or input_ids.shape[-1] == 1:
                input_ids = None
                attention_mask = None

        # print(time.time(), 'start forward')
        if self.no_text:
            with torch.no_grad():
                if pixel_values is not None:
                    vision_outputs = self.vision_model(
                        pixel_values=pixel_values,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

                    image_embeds = vision_outputs[1]
                    image_embeds = self.visual_projection(image_embeds)

                    # normalized features
                    image_embeds = image_embeds / _get_vector_norm(image_embeds)
                    
        else:
            if pixel_values is not None:
                vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                image_embeds = vision_outputs[1]
                image_embeds = self.visual_projection(image_embeds)

                # normalized features
                image_embeds = image_embeds / _get_vector_norm(image_embeds)
        
        # print(time.time(), 'ok vision')

        if input_ids is not None:

            with torch.no_grad():
                text_outputs = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                text_embeds = text_outputs[1]
                text_embeds = self.text_projection(text_embeds)
                
                text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # print(time.time(), 'ok text')

        touch_embeds = self.get_touch_features(touch_input, sensor_type=sensor_type, data_type = data_type)
        touch_embeds = touch_embeds / _get_vector_norm(touch_embeds)

        # print(touch_embeds.shape, text_embeds.shape, image_embeds.shape)
        # print(time.time(), 'ok touch')

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        logits_per_touch_image = None
        logits_per_text_image = None
        logits_per_touch_text= None

        if (touch_embeds is not None) and (image_embeds is not None):
            logits_per_touch_image = torch.matmul(touch_embeds[vision_indices], image_embeds.t().to(touch_embeds.device)) * logit_scale.to(
                touch_embeds.device
            )
            #logits_per_image_touch = logits_per_touch_image.t()

        if (text_embeds is not None) and (image_embeds is not None):
            if vision_indices_t is not None and vision_indices_t.shape[0] > 0:
                logits_per_text_image = torch.matmul(text_embeds[text_indices_v], image_embeds[vision_indices_t].t().to(text_embeds.device)) * logit_scale.to(
                    text_embeds.device
                )
            #logits_per_image_text = logits_per_text_image.t()

        if (touch_embeds is not None) and (text_embeds is not None):
            logits_per_touch_text = torch.matmul(touch_embeds[text_indices], text_embeds.t().to(touch_embeds.device)) * logit_scale.to(
                touch_embeds.device
            )
            #logits_per_text_touch = logits_per_touch_text.t()
        # loss = None
        # if return_loss:
        mae_loss = torch.zeros(1)

        # print(logits_per_touch_image.shape if logits_per_touch_image is not None else None, logits_per_text_image.shape if logits_per_text_image is not None else None, logits_per_touch_text.shape if logits_per_touch_text is not None else None)

        align_loss = self.clip_loss(logits_per_touch_image, logits_per_text_image, logits_per_touch_text)

        # print(time.time(), 'ok align loss')

        if self.do_mae:
            mae_loss,_,_ = self.touch_mae_model(touch_input, sensor_type=sensor_type, data_type=data_type, target_sensor_type = target_sensor_type)

        # print(logits_per_touch_image.shape, logits_per_text_image.shape, logits_per_touch_text.shape)
        # print(time.time(), 'ok mae loss')
        return align_loss, mae_loss

        # if not return_dict:
        #     output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        #     return ((loss,) + output) if loss is not None else output

        # return CLIPOutput(
        #     loss=loss,
        #     logits_per_image=logits_per_image,
        #     logits_per_text=logits_per_text,
        #     text_embeds=text_embeds,
        #     image_embeds=image_embeds,
        #     text_model_output=text_outputs,
        #     vision_model_output=vision_outputs,
        # )

