#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/06/28 14:45:20
@email: fjjth98@163.com
@description: Evaluate Video-CCAM-4B on Video-MME Benchmark
================================================
"""

import os
import json
import torch
import os.path as osp

from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from eval import video_collate_fn, VideoMMEDataset
from model import create_videoccam, DEFAULT_VIDEO_TOKEN, VideoCCAM


def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--model_path', type=str, help='Local model path')
    parser.add_argument('--llm_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='LLM model name or path')
    parser.add_argument('--visual_encoder_name_or_path', type=str, default='google/siglip-so400m-patch14-384', help='Visual encoder model name or path')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Model data type, only support `float32`, `float16`, and `bfloat16`')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--benchmark', type=str, choices=['Video-MME'], help='Supported benchmarks')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    return parser.parse_args()


@torch.inference_mode
def eval_videomme(
    mllm: VideoCCAM,
    dataset_path: str,
    output_dir: str,
    sample_config: dict,
    batch_size: int = 4,
):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    dataset = VideoMMEDataset(
        dataset_path=dataset_path,
        sample_config=sample_config
    )
    for use_subtitle in (False, True):
        dataset.use_subtitle = use_subtitle
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=video_collate_fn
        )
        results = []
        for data in tqdm(dataloader):
            response, pixel_values = mllm.generate(
                texts=['\n'.join([DEFAULT_VIDEO_TOKEN, t]) for t in data['text'][0]],
                videos=data['video'],
                return_pixel_values=True
            )
            response = [response]
            for i in range(1, len(data['text'])):
                response.append(mllm.generate(
                    texts=['\n'.join([DEFAULT_VIDEO_TOKEN, t]) for t in data['text'][i]],
                    pixel_values=pixel_values
                ))
            response = [[response[i][j] for i in range(len(response))] for j in range(len(response[0]))]
            results.extend(response)

        outputs = []
        for data, responses in zip(dataset.data_list, results):
            data = deepcopy(data)
            data.pop('videoID')
            for question, response in zip(data['questions'], responses):
                question['response'] = response
            outputs.append(data)

        suffix = 'w_sub' if use_subtitle else 'wo_sub'
        with open(osp.join(output_dir, f'output_{suffix}.json'), 'w') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    args = parse_args()

    mllm = create_videoccam(
        model_name=args.model_name,
        model_path=args.model_path,
        llm_name_or_path=args.llm_name_or_path,
        visual_encoder_name_or_path=args.visual_encoder_name_or_path,
        torch_dtype=eval('torch.' + args.dtype)
    )

    sample_config=dict(
        sample_type='uniform',
        num_frames=args.num_frames
    )

    if args.benchmark == 'Video-MME':
        eval_videomme(
            mllm=mllm,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sample_config=sample_config,
            batch_size=args.batch_size,
        )
