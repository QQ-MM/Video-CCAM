#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/06/28 14:28:30
@email: fjjth98@163.com
@description: Benchmark datasets
================================================
"""

import os
import cv2
import json
import imageio
import numpy as np
import os.path as osp

from PIL import Image
from pandas import read_parquet
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from .utils import load_decord


class VideoMMEDataset(Dataset):
    """Video-MME dataset. By default, videos are saved in `video/` and subtitles are saved in `subtitle/`
    """

    def __init__(self, dataset_path: str, sample_config: dict, use_subtitle: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.sample_config = sample_config
        self.use_subtitle = use_subtitle

        data_dict = {}
        index_keys = ['video_id', 'duration', 'domain', 'sub_category', 'videoID']
        value_keys = ['question_id', 'task_type', 'question', 'options', 'answer']
        df = read_parquet(osp.join(dataset_path, 'videomme', 'test-00000-of-00001.parquet'))
        df['options'] = df['options'].apply(list)
        for _, data in df.iterrows():
            key = tuple(data[k] for k in index_keys)
            value = data[value_keys].to_dict()
            if key in data_dict:
                data_dict[key].append(value)
            else:
                data_dict[key] = [value]
        self.data_list = [dict(zip(index_keys + ['questions'], list(k) + [v])) for k, v in data_dict.items()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> dict:
        if self.use_subtitle:
            frames, subtitles = load_decord(
                src_path=osp.join(self.dataset_path, 'video', self.data_list[idx]['videoID'] + '.mp4'),
                sub_path=osp.join(self.dataset_path, 'subtitle', self.data_list[idx]['videoID'] + '.srt'),
                **self.sample_config
            )
            text = ['\n'.join([
                "This video's subtitles are listed below:",
                subtitles,
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.data_list[idx]['questions']]
        else:
            frames = load_decord(
                src_path=osp.join(self.dataset_path, 'video', self.data_list[idx]['videoID'] + '.mp4'),
                **self.sample_config
            )
            text = ['\n'.join([
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.data_list[idx]['questions']]
            subtitles = ''

        return dict(
            video=frames,
            text=text
        )


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

    def __init__(self, dataset_path: str, sample_config: int = 16, task_types: list[str] = None):
        """
        Args:
            dataset_path (str): MVBench dataset path (directory)
            transform (Callable): image_processor
            num_frame (int, optional): Number of frames. Defaults to 16.
            task_types (list[str], optional): (Optional) Task types. Defaults to None (All tasks are evaluated).
        """
        super().__init__()
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

        self.num_frame = sample_config['num_frames']

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
        seg_size = float(end_idx - start_idx) / self.num_frame
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_frame)
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
        for k in ('Answer:', 'The answer is'):
            predict = predict.removeprefix(k).strip()
        predict_option = predict.split(' ')[0].lower().replace('.', '')
        answer_option = answer.split(' ')[0].lower()
        return len(predict_option) > 0 and (predict_option in answer_option or answer_option in predict_option)


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
