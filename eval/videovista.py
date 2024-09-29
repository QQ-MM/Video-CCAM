#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/08/22 19:38:43
@email: fjjth98@163.com
@description: VideoVista Evaluation
================================================
"""
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import load_decord, video_collate_fn


class VideoVistaDataset(Dataset):

    understanding = [
        "Objects Existence", "Objects Count", "Action Count", "Detailed Description", 'Brief Description', 'Event Description', 'Event Sequence', 'Optical Character Recognition',
        'Action Recognition',  'Action Sequence', 'Action Location', 'Event Location', 'Objects Temporal Location', 'Objects Temporal Relation',
        'Objects Spatial Location', 'Objects Spatial Relation', 'Objects Spatial Tracking', 'Human Activity Analysis', 'Anomaly Detection'
    ]

    reasoning = [
        'Relation Reasoning-Image', 'Relation Reasoning-Video', 'Event Prediction', 'Action Prediction', 'Causal Reasoning', 'Counterfactual Reasoning', 'Commonsense Reasoning', 'Logic Reasoning'
    ]

    def __init__(self, dataset_path: str, sample_config: dict) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.sample_config = sample_config

        with open(osp.join(dataset_path, 'VideoVista.json'), 'r') as f:
            self.data_list = json.load(f)
        for item in tqdm(self.data_list):
            duration_group = item['video_name'].split('.')[1]
            if osp.exists(osp.join(self.dataset_path, 'merged', item['category'], duration_group, item['video_name'])):
                item['video'] = osp.join(self.dataset_path, 'merged', item['category'], duration_group, item['video_name'])
            else:
                for i in os.listdir(osp.join(self.dataset_path, 'merged')):
                    item['video'] = osp.join(self.dataset_path, 'merged', i, duration_group, item['video_name'])
                    if osp.exists(item['video']):
                        break
                else:
                    print(item['video_name'], ' not exist')
                    raise ValueError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) -> dict:
        item = self.data_list[index]

        # find actual video path and load video
        video = load_decord(item['video'], **self.sample_config)

        # wrap choices into question
        question = item['Question'] + '\nOptions:\n'
        for idx, option in enumerate(item['Answer_Choices']):
            option = f"({chr(ord('A') + idx)}) {option}\n"
            question += option
            if idx == item['Answer']:
                answer = option

        return dict(
            video=video,
            question=question,
            answer=answer,
            task_type=item['Type']
        )

    @staticmethod
    def check_answer(predict: str, answer: str) -> bool:
        for k in ('Answer:', 'The answer is', 'Answer is'):
            predict = predict.removeprefix(k).strip()
        predict_option = predict.split(' ')[0].lower().replace('.', '')
        answer_option = answer.split(' ')[0].lower()
        return len(predict_option) > 0 and (predict_option in answer_option or answer_option in predict_option)


@torch.inference_mode
def evaluate(
    model,
    tokenizer,
    image_processor,
    dataset_path: str,
    output_path: str,
    sample_config: dict,
    batch_size: int = 4,
    question_prompt: str = "Answer with the option's letter from the given choices directly."
):
    if not osp.exists(output_path):
        os.makedirs(output_path)

    dataset = VideoVistaDataset(
        dataset_path=dataset_path,
        sample_config=sample_config
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=video_collate_fn
    )

    # Get raw results
    results = {k: [] for k in dataset.understanding + dataset.reasoning}
    for data in tqdm(dataloader):
        messages = [[{
            'role': 'user',
            'content': f'<video>\n{question}\n{question_prompt}'
        }] for question in data['question']]
        images = data['video']
        response = model.chat(messages, images, tokenizer, image_processor, max_new_tokens=100, do_sample=False)
        for answer, task_type, predict in zip(data['answer'], data['task_type'], response):
            results[task_type].append(dict(
                predict=predict,
                answer=answer,
                correct=dataset.check_answer(predict, answer)
            ))
    with open(osp.join(output_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    accuracy, correct, total = {}, {}, {}
    for k, v in results.items():
        correct[k] = len(list(filter(lambda x: x['correct'], v)))
        total[k] = len(v)
        accuracy[k] = round(correct[k] / total[k] * 100, 2)

    understanding_correct = sum(correct[k] for k in VideoVistaDataset.understanding)
    understanding_total = sum(total[k] for k in VideoVistaDataset.understanding)
    accuracy['understanding'] = round(understanding_correct / understanding_total * 100, 2)

    reasoning_correct = sum(correct[k] for k in VideoVistaDataset.reasoning)
    reasoning_total = sum(total[k] for k in VideoVistaDataset.reasoning)
    accuracy['reasoning'] = round(reasoning_correct / reasoning_total * 100, 2)

    accuracy['avg'] = round((understanding_correct + reasoning_correct) / (understanding_total + reasoning_total) * 100, 2)
    print(f'Understanding: {accuracy["understanding"]}%, Reasoning: {accuracy["reasoning"]}%, Avg: {accuracy["avg"]}%')

    with open(osp.join(output_path, 'upload_leaderboard.json'), 'w') as f:
        json.dump(accuracy, f, indent=4, ensure_ascii=False)
