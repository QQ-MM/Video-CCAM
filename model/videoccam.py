#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/06/23 09:52:24
@email: fjjth98@163.com
@description:
================================================
"""

import torch
import os.path as osp
import torch.nn as nn

from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel, SiglipImageProcessor


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_VIDEO_TOKEN = '<video>'


class VideoCCAM(nn.Module):

    def __init__(
        self,
        model_path: str,
        chat_template: str,
        generation_args: dict,
        llm_name_or_path: str = None,
        visual_encoder_name_or_path: str = None,
        special_tokens: list[str] = None,
        visual_select_layer: int = -2,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = 'cuda:0'
    ):
        super().__init__()
        self.chat_template = chat_template
        self.generation_args = generation_args
        self.visual_select_layer = visual_select_layer
        self.torch_dtype = torch_dtype
        self.device_map = device_map

        if llm_name_or_path is None:
            llm_name_or_path = model_path
        if visual_encoder_name_or_path is None:
            visual_encoder_name_or_path = osp.join(model_path, 'visual_encoder')
            assert osp.exists(visual_encoder_name_or_path), f'{visual_encoder_name_or_path} does not exist, you have to specify `visual_encoder_name_or_path`'
        projector_path = osp.join(model_path, 'projector')
        assert osp.exists(projector_path), f'{projector_path} does not exist, you have to change `model_path`'

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name_or_path,
            trust_remote_code=True
        )
        print(f'Load LLM from {llm_name_or_path}')
        if special_tokens is not None:
            self.llm.resize_token_embeddings(self.llm.get_input_embeddings().weight.size(0) + len(special_tokens))
            self.llm.requires_grad_(False)
            self.llm.get_input_embeddings().weight[-len(special_tokens):].zero_()
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
            print(f'Add special_tokens {special_tokens} to LLM and tokenizer')
        if osp.exists(adapter_path := osp.join(model_path, 'llm_adapter')):
            self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
            print(f'Load LLM adapter from {adapter_path}')
        self.generation_args['eos_token_id'] = self.tokenizer.convert_tokens_to_ids(self.generation_args.pop('stop_tokens'))

        self.visual_encoder = SiglipVisionModel.from_pretrained(
            visual_encoder_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(visual_encoder_name_or_path)
        print(f'Load SigLIP visual encoder from {visual_encoder_name_or_path}')
        if osp.exists(adapter_path := osp.join(model_path, 'visual_encoder_adapter')):
            self.visual_encoder = PeftModel.from_pretrained(self.visual_encoder, adapter_path)
            print(f'Load visual_encoder adapter from {adapter_path}')

        self.projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        print(f'Load projector from {projector_path}')

    # Modified from https://github.com/InternLM/xtuner/blob/main/xtuner/model/utils.py#L138
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        labels: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None
    ):
        if pixel_values is None:
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'inputs_embeds': None,
                'labels': labels
            }

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            if isinstance(input_ids, torch.Tensor):
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            elif isinstance(input_ids, list):
                attention_mask = [torch.ones_like(i, dtype=torch.bool) for i in input_ids]
                _attention_mask = attention_mask
            else:
                raise ValueError(f'Do not support {type(input_ids)} type as input_ids')
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids[0].shape[0], dtype=torch.long, device=input_ids[0].device)
        if labels is None:
            if isinstance(input_ids, torch.Tensor):
                labels = torch.full_like(input_ids, IGNORE_INDEX)
            elif isinstance(input_ids, list):
                labels = [torch.full_like(i, IGNORE_INDEX) for i in input_ids]
            else:
                raise ValueError(f'Do not support {type(input_ids)} type as input_ids')

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_inputs_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_pixel_values = pixel_values[cur_image_idx]
                cur_inputs_embeds_1 = self.llm.get_input_embeddings()(cur_input_ids)
                cur_inputs_embeds = torch.cat(
                    [cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
                new_inputs_embeds.append(cur_inputs_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(
                cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                    cur_input_ids.shape[0]
                ]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                        1:image_token_indices[i +
                                                                            1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] +
                                                1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_inputs_embeds = self.llm.get_input_embeddings()(
                torch.cat(cur_input_ids_noim))
            cur_inputs_embeds_no_im = torch.split(
                cur_inputs_embeds, split_sizes, dim=0)
            cur_new_inputs_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_pixel_values = pixel_values[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_inputs_embeds.append(cur_pixel_values)
                    cur_new_labels.append(
                        torch.full((cur_pixel_values.shape[0], ),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype))

            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_inputs_embeds.append(cur_new_inputs_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_inputs_embeds)
        batch_size = len(new_inputs_embeds)

        new_inputs_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len),
                                    IGNORE_INDEX,
                                    dtype=new_labels[0].dtype,
                                    device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len),
                                    dtype=attention_mask[0].dtype,
                                    device=attention_mask[0].device)
        position_ids = torch.zeros((batch_size, max_len),
                                dtype=position_ids.dtype,
                                device=position_ids.device)

        for i, (cur_new_embed,
                cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_inputs_embeds_padded.append(
                torch.cat((cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                    dtype=cur_new_embed.dtype,
                                    device=cur_new_embed.device)),
                        dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0,
                    cur_len,
                    dtype=position_ids.dtype,
                    device=position_ids.device)

        new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        elif isinstance(_attention_mask, list):
            attention_mask = attention_mask.to(dtype=_attention_mask[0].dtype)
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_inputs_embeds,
            'labels': new_labels
        }

    def generate(
        self,
        texts: list[str],
        videos: list[list[Image.Image]] = None,
        pixel_values: torch.Tensor = None,
        return_pixel_values: bool = False
    ) -> list[str]:
        """Genrate respoonse for video and text inputs.

        Args:
            text (list[str]): list of text inputs
            video (list[list[Image.Image]], optional): list of frame list. Defaults to None.
            pixel_values (torch.Tensor, optional): precomputed pixel_values. Defaults to None.
            return_pixel_values (bool, optional): whether return pixel values or not. Defaults to False.

        Returns:
            list[str]: _description_
        """
        prediction = []
        # Get visual embeddings
        if pixel_values is None:
            frames, split_sizes = [], []
            for i in videos:
                frames += i
                split_sizes.append(len(i))
            pixel_values = self.image_processor(frames, return_tensors='pt')['pixel_values'].to(self.torch_dtype).to(self.device_map)
            pixel_values = self.visual_encoder(pixel_values, output_hidden_states=True).hidden_states[self.visual_select_layer]
            pixel_values = self.projector(pixel_values, split_sizes)

        for i, t in enumerate(texts):
            et = self.chat_template.format(input=t).replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN).split(DEFAULT_IMAGE_TOKEN)
            assert len(et) == 2, f'Wrong input formats for {t}'
            input_ids = [torch.tensor(self.tokenizer.encode(et[0]) + [IMAGE_TOKEN_INDEX] + self.tokenizer.encode(et[1], add_special_tokens=False), device=self.device_map)]
            mm_inputs = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                pixel_values=pixel_values[i:i+1]
            )
            generate_output = self.llm.generate(
                **mm_inputs,
                **self.generation_args
            )[0]
            prediction.append(self.tokenizer.decode(generate_output, skip_special_tokens=True))

        if return_pixel_values:
            return prediction, pixel_values
        else:
            return prediction
