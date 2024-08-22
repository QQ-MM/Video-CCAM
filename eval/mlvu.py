#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/08/22 15:25:11
@email: fjjth98@163.com
@description: MLVU Evaluation
================================================
"""
import os
import json
import torch
import os.path as osp

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .utils import load_decord, video_collate_fn


class MLVUDataset(Dataset):

    _raw_data = dict(
        M={
            'count': ('4_count.json', 'video/count', 'video'),
            'ego': ('3_ego.json', 'video/ego', 'video'),
            'needle': ('2_needle.json', 'video/needle', 'video'),
            'order': ('5_order.json', 'video/order', 'video'),
            'plotQA': ('1_plotQA.json', 'video/plotQA', 'video'),
            'anomaly_reco': ('6_anomaly_reco.json', 'video/anomaly_reco', 'video'),
            'topic_reasoning': ('7_topic_reasoning.json', 'video/topic_reasoning', 'video')
        },
        G={
            'subPlot': ('8_sub_scene.json', 'video/subPlot', 'video'),
            'summary': ('9_summary.json', 'video/summary', 'video')
        }
    )

    def __init__(self, dataset_path: str, task_name: str, sample_config: dict):
        """
        Args:
            dataset_path (str): _description_
            task_name (str): ['M', 'G'], 'multiple choice', 'generation'
            sample_config (dict): _description_
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.sample_config = sample_config

        self.data_list = []
        for k, (json_path, video_path, data_type) in self._raw_data[self.task_name].items():
            prefix = osp.join(dataset_path, 'video', osp.splitext(json_path)[0])
            with open(osp.join(dataset_path, 'json', json_path), 'r') as f:
                for data in json.load(f):
                    self.data_list.append({
                        'task_type': k,
                        'prefix': prefix,
                        'data_type': data_type,
                        'data': data
                    })

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])

        correct = 0
        total = 0
        res = f'There are {len(self.data_list)} videos as follow:\n'
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f'{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n'
            correct = correct + 1 / option_list[k]
        res += f'Total random accuracy: {correct/total*100:.2f}%'
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def qa_template(self, data):
        if self.task_name == 'M':
            question = f"Question: {data['question']}\n"
            question += 'Options:\n'
            answer = data['answer']
            answer_idx = -1
            for idx, c in enumerate(data['candidates']):
                question += f"({chr(ord('A') + idx)}) {c}\n"
                if c == answer:
                    answer_idx = idx
            question = question.rstrip()
            answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        else:
            question = data['question']
            answer = data['answer']
        return question, answer

    def __getitem__(self, idx):
        video_path = osp.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        images = load_decord(video_path, **self.sample_config)
        question, answer = self.qa_template(self.data_list[idx]['data'])

        return {
            'video_name': osp.split(self.data_list[idx]['data']['video'])[1],
            'video': images,
            'question': question,
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

    @staticmethod
    def check_answer(predict: str, answer: str) -> bool:
        for k in ('Answer:', 'The answer is'):
            predict = predict.removeprefix(k).strip()
        predict_option = predict.split(' ')[0].lower().replace('.', '')
        answer_option = answer.split(' ')[0].lower()
        return len(predict_option) > 0 and (predict_option in answer_option or answer_option in predict_option)


@torch.inference_mode
def evaluate(
    model,
    dataset_path: str,
    output_dir: str,
    sample_config: dict,
    batch_size: int = 4,
    question_prompt: str = "Answer with the option's letter from the given choices directly."
):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Multiple choice
    dataset = MLVUDataset(
        dataset_path=dataset_path,
        task_name='M',
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
    results = {k: [] for k in dataset._raw_data['M']}
    for data in tqdm(dataloader):
        messages = [[{
            'role': 'user',
            'content': f'<video>\n{question}\n{question_prompt}'
        }] for question in data['question']]
        images = data['video']
        response = model.chat(messages, images, max_new_tokens=100, do_sample=False)
        for answer, task_type, predict in zip(data['answer'], data['task_type'], response):
            results[task_type].append(dict(
                predict=predict,
                answer=answer,
                correct=dataset.check_answer(predict, answer)
            ))
    with open(osp.join(output_dir, 'choice_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    # Get accuracy
    accuracy, correct_count, total_count = {}, 0, 0
    for k, v in results.items():
        correct = len(list(filter(lambda x: x['correct'], v)))
        total = len(v)
        accuracy[k] = round(correct / total * 100, 2)
        correct_count += correct
        total_count += total
    accuracy['Avg'] = round(correct_count / total_count * 100 + 1e-5, 2)    # correct rounding 55.125 -> 55.13
    print(f'Total accuracy: {accuracy["Avg"]}%')
    with open(osp.join(output_dir, 'choice_leaderboard.json'), 'w') as f:
        json.dump(accuracy, f, indent=4, ensure_ascii=False)

    # Generation
    dataset = MLVUDataset(
        dataset_path=dataset_path,
        task_name='G',
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
    results = {k: [] for k in dataset._raw_data['G']}
    for data in tqdm(dataloader):
        messages = [[{
            'role': 'user',
            'content': f'<video>\n{question}'
        }] for question in data['question']]
        images = data['video']
        response = model.chat(messages, images, max_new_tokens=512, num_beams=5, do_sample=False)
        for video_name, question, answer, predict, task_type in zip(data['video_name'], data['question'], data['answer'], response, data['task_type']):
            results[task_type].append(dict(
                video_name=video_name,
                Q=question,
                A=answer,
                pred=predict
            ))
    for task_type, results in results.items():
        with open(osp.join(output_dir, f'generation_{task_type}_results.json'), 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
