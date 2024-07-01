#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/06/28 15:14:43
@email: fjjth98@163.com
@description: 
================================================
"""

import torch
import os.path as osp

from .videoccam import VideoCCAM


def _create_videoccam(
    model_path: str,
    llm_name_or_path: str, 
    chat_template: str,
    stop_words: list[str],
    visual_encoder_name_or_path: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    special_tokens: list[str] = None,
    visual_select_layer: int = -2,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda:0'
) -> VideoCCAM:
    if isinstance(torch_dtype, str):
        torch_dtype = eval('torch.' + torch_dtype)

    mllm = VideoCCAM(
        model_path=model_path,
        chat_template=chat_template,
        generation_args=dict(
            stop_tokens=stop_words,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        ),
        llm_name_or_path=llm_name_or_path,
        visual_encoder_name_or_path=visual_encoder_name_or_path,
        special_tokens=special_tokens,
        visual_select_layer=visual_select_layer,
        torch_dtype=torch_dtype,
        device_map=device
    )

    return mllm.eval()


def create_videoccam(model_name: str = None, **kwargs) -> VideoCCAM:
    
    if model_name is None:
        model_name = osp.split(kwargs['model_path'])
        print(f'Guess `model_name` as {model_name}')

    if model_name == 'Video-CCAM-4B':
        kwargs['chat_template'] = '<|user|>\n{input}<|end|>\n<|assistant|>\n'
        kwargs['stop_words'] = ['<|end|>', '<|endoftext|>']
        if kwargs.get('llm_name_or_path') is None:
            kwargs['llm_name_or_path'] = 'microsoft/Phi-3-mini-4k-instruct'
        mllm = _create_videoccam(**kwargs)
    elif model_name == 'Video-CCAM-9B':
        kwargs['chat_template'] = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
        kwargs['stop_words'] = ['<|im_end|>']
        if kwargs.get('llm_name_or_path') is None:
            kwargs['llm_name_or_path'] = '01-ai/Yi-1.5-9B-Chat'
        mllm = _create_videoccam(**kwargs)
    else:
        raise NotImplementedError(f'Do not support {model_name}, currently only support ["Video-CCAM-4B", "Video-CCAM-9B"]')

    return mllm
