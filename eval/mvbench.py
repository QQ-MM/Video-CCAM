#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/08/22 14:41:26
@email: fjjth98@163.com
@description: MVBench Evaluation
================================================
"""
import json
import os
import os.path as osp

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import video_collate_fn


# Modified from: https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb
class MVBenchDataset(Dataset):

    _raw_data = {
        "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", "perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", "perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
        "Character Order": ("character_order.json", "perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
    }

    def __init__(self, dataset_path: str, num_frames: int = 32, task_types: list[str] = None):
        """_summary_

        Args:
            dataset_path (str): _description_
            num_frames (int, optional): _description_. Defaults to 32.
            task_types (list[str], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.num_frames = num_frames
        self.data_list = []
        for k, v in self._raw_data.items():
            if isinstance(task_types, list) and k not in task_types:
                continue
            prefix = osp.join(dataset_path, 'video', v[1])
            with open(osp.join(dataset_path, 'json', v[0]), 'r') as f:
                for data in json.load(f):
                    self.data_list.append({
                        'task_type': k,
                        'prefix': prefix,
                        'data_type': v[2],
                        'bound': v[3],
                        'data': data
                    })

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

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
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_frames
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_frames)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        return images_group

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        return images_group

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(osp.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        return images_group

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = osp.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])

        return {
            'video': torch_imgs,
            'question': question,
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

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
    num_frames: int = 32,
    batch_size: int = 4,
    question_prompt: str = "Answer with the option's letter from the given choices directly."
):
    if not osp.exists(output_path):
        os.makedirs(output_path)

    dataset = MVBenchDataset(
        dataset_path=dataset_path,
        num_frames=num_frames
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
    results = {k: [] for k in dataset._raw_data}
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
    with open(osp.join(output_path, 'upload_leaderboard.json'), 'w') as f:
        json.dump(accuracy, f, indent=4, ensure_ascii=False)
