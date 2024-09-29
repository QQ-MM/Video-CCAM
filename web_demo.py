#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/09/24 19:50:20
@email: fjjth98@163.com
@description:
================================================
"""
import os
from argparse import ArgumentParser

import gradio as gr
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import (AutoImageProcessor, AutoModel, AutoTokenizer,
                          DynamicCache)


def sample_frame_idx(sample_type: str, end_idx: int, start_idx: int = 0, **kwargs) -> list[int]:
    """Sample index from sequence [start_idx, end_idx)

    Args:
        sample_type (str): 'rand', 'uniform', 'fps'
        end_idx (int): total number of frames

    Returns:
        list[int]: sampled frame index
    """
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        if num_frames > end_idx - start_idx:
            frame_idx = list(range(start_idx, end_idx))
        else:
            splits = torch.linspace(start_idx, end_idx, num_frames+1)
            frame_idx = ((splits[:-1] + splits[1:]) // 2).int().tolist()
    elif sample_type == 'fps':
        input_fps = kwargs.pop('input_fps')
        max_num_frames = kwargs.pop('max_num_frames', -1)
        delta = input_fps / kwargs.pop('output_fps') if 'output_fps' in kwargs else 1
        if delta <= 1:
            frame_idx = list(range(start_idx, end_idx))
        else:
            frame_idx = torch.arange(start_idx, end_idx, delta).int().tolist()
        if 0 < max_num_frames < len(frame_idx):
            frame_idx = sample_frame_idx('uniform', end_idx, start_idx, num_frames=max_num_frames)
    else:
        raise ValueError(f'Do not support sample_type as {sample_type}.')
    return frame_idx


def load_video(video_path: str, **kwargs) -> list[Image.Image]:
    """Load video using decord

    Args:
        video_path (str): video path

    Returns:
        list[Image.Image]: image list
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    kwargs['end_idx'] = len(vr)
    if kwargs['sample_type'] == 'fps':
        kwargs['input_fps'] = float(vr.get_avg_fps())
    idx = sample_frame_idx(**kwargs)
    frames = [Image.fromarray(frame).convert('RGB') for frame in vr.get_batch(idx).asnumpy()]
    return frames


def get_avg_fps_gif(im: Image.Image, default_fps: int = 20) -> float:
    """Get average fps for gif

    Args:
        im (Image.Image, optional): gif Image object from video_path. Defaults to None.
        default_fps (int, optional): default fps if no duration is read. Defaults to 20.

    Returns:
        float: average fps
    """
    duration, count = 0., 0
    for i in range(im.n_frames):
        try:
            im.seek(i)
            duration += im.info['duration']     # some frame do not have duration
            count += 1
        except:
            pass
    fps = default_fps if count == 0 else 1000 * count / duration
    return fps


def load_gif(video_path: str, **kwargs) -> list[Image.Image]:
    """Load gif using PIL.Image

    Args:
        video_path (str): gif path

    Returns:
        list[Image.Image]: Image sequence list
    """
    frames = []
    with Image.open(video_path) as im:
        kwargs['end_idx'] = im.n_frames
        if kwargs['sample_type'] == 'fps':
            kwargs['input_fps'] = get_avg_fps_gif(im)
        idx = sample_frame_idx(**kwargs)
        for i in idx:
            im.seek(i)
            frames.append(im.convert('RGB'))
    return frames


def load_frames(video_path: str, **kwargs) -> list[Image.Image]:
    """Load video from frame directory

    Args:
        video_path (str): frame directory,

    Returns:
        list[Image.Image]: image list
    """
    assert kwargs['sample_type'] != 'fps', f'Do not support loading frames by fps'
    # WARNING: frame names must be able to be sorted in order!
    names = os.listdir(video_path)
    names.sort()
    kwargs['end_idx'] = len(names)
    idx = sample_frame_idx(**kwargs)
    frames = []
    for i in idx:
        with Image.open(os.path.join(video_path, names[i])) as im:
            frames.append(im.convert('RGB'))
    return frames


def load_image_or_video(vision_path: str, **kwargs) -> list[Image.Image]:
    ext = os.path.splitext(vision_path)[1]
    if ext == '.gif':
        images = load_gif(vision_path, **kwargs)
    elif ext in {'.jpeg', '.jpg', '.png'}:
        with Image.open(vision_path) as im:
            images = [im.convert('RGB')]
    elif os.path.isdir(vision_path):
        images = load_frames(vision_path, **kwargs)
    else:
        images = load_video(vision_path, **kwargs)
    return images
    

# parse arguments
parser = ArgumentParser(description='demo')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--torch_dtype', type=str, default='bfloat16')
parser.add_argument('--device_map', type=str, default='auto')
parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_path)
image_processor = AutoImageProcessor.from_pretrained(args.model_path)
videoccam = AutoModel.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype=eval('torch.' + args.torch_dtype),
    device_map=args.device_map,
    attn_implementation=args.attn_implementation
)
past_key_values = DynamicCache()

def add_message(message: dict, history: list[tuple[str, str]]):
    for file in message['files']:
        history.append(((file,), None))
    history.append((message['text'], None))
    return gr.MultimodalTextbox(interactive=False), history


def chat(
    message: dict,
    history: list[tuple[str, str]],
    # generation_config
    do_sample: bool,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    # video sampling config,
    sample_type: str,
    num_frames: int,
    max_num_frames: int,
    fps: float
):
    pixel_values, vision_split_sizes = [], []
    for file in message['files']:
        cur_pixel_values = load_image_or_video(
            file,
            sample_type=sample_type,
            num_frames=num_frames,
            max_num_frames=max_num_frames,
            output_fps=fps
        )
        pixel_values += cur_pixel_values
        vision_split_sizes.append(len(cur_pixel_values))
    vision_count = message['text'].count('<image>') + message['text'].count('<video>')
    if vision_count < len(vision_split_sizes):
        message['text'] = '<image>' * (len(vision_split_sizes) - vision_count) + message['text']
    input_ids = tokenizer.apply_chat_template(
        [dict(role='user', content=message['text'])],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False
    )
    if len(pixel_values) > 0:
        pixel_values = image_processor(pixel_values, return_tensors='pt')['pixel_values'].to(
            dtype=videoccam.vision_encoder.get_input_embeddings().weight.dtype,
            device=videoccam.vision_encoder.get_input_embeddings().weight.device
        )
    else:
        pixel_values = None
    global past_key_values
    output_ids, past_key_values = videoccam.generate(
        input_ids=[input_ids],
        pixel_values=pixel_values,
        vision_split_sizes=vision_split_sizes,
        past_key_values=past_key_values,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    history[-1][1] = tokenizer.decode(output_ids, skip_special_tokens=True)
    return gr.MultimodalTextbox(value=None, interactive=True), history


def clear_cache():
    global past_key_values
    past_key_values = DynamicCache()
    return gr.MultimodalTextbox(value=None, interactive=True), None


with gr.Blocks(title='Video-CCAM') as chat_interface:
    gr.Markdown('# ' + os.path.basename(args.model_path))
    with gr.Row():
        chatbot = gr.Chatbot(height=800, scale=50)
        with gr.Column(scale=1):
            with gr.Accordion(label='Generation'):
                generation_config=dict(
                    do_sample=gr.Checkbox(
                        videoccam.generation_config.do_sample,
                        interactive=True,
                        label='do_sample'
                    ),
                    max_new_tokens=gr.Number(
                        1024,
                        precision=0,
                        interactive=True,
                        label='max_new_tokens'
                    ),
                    num_beams=gr.Number(
                        videoccam.generation_config.num_beams,
                        precision=0,
                        interactive=True,
                        label='num_beams'
                    ),
                    temperature=gr.Number(
                        videoccam.generation_config.temperature,
                        interactive=True,
                        label='temperature'
                    ),
                    top_k=gr.Number(
                        videoccam.generation_config.top_k,
                        precision=0,
                        interactive=True,
                        label='top_k'
                    ),
                    top_p=gr.Number(
                        videoccam.generation_config.top_p,
                        interactive=True,
                        label='top_p'
                    ),
                    repetition_penalty=gr.Number(
                        videoccam.generation_config.repetition_penalty,
                        interactive=True,
                        label='repetition_penalty'
                    )
                )
            with gr.Accordion(label='Video Sampling', open=False):
                sample_config=dict(
                    sample_type=gr.Radio(
                        choices=['uniform', 'fps'],
                        interactive=True,
                        value='uniform',
                        label='sample_type'
                    ),
                    num_frames=gr.Slider(
                        1, 512,
                        interactive=True,
                        value=32,
                        label='num_frames'
                    ),
                    max_num_frames=gr.Number(
                        -1,
                        interactive=True,
                        label='max_num_frames'
                    ),
                    fps=gr.Number(
                        1,
                        interactive=True,
                        label='fps'
                    )
                )
    with gr.Row():
        mm_input = gr.MultimodalTextbox(
            interactive=True,
            file_count='multiple',
            placeholder='Enter message or upload file...',
            show_label=False,
            scale=50
        )
        mm_input.submit(
            add_message, [mm_input, chatbot], [mm_input, chatbot]
        ).then(
            chat,
            [mm_input, chatbot] + list(generation_config.values()) + list(sample_config.values()),
            [mm_input, chatbot]
        )
        clear = gr.Button(value='🗑️  Clear', scale=1)
        clear.click(clear_cache, None, [mm_input, chatbot])
    gr.Examples(
        examples=[
            {'text': '中国的首都是哪里？', 'files': []},
            {'text': 'When is the National Day in the United States?', 'files': []},
            {'text': 'Please describe this image in detail.', 'files': ['assets/example_image.jpg']},
            {'text': '请仔细描述这个视频。', 'files': ['assets/example_video.mp4']},
        ],
        inputs=mm_input
    )

chat_interface.launch(server_port=8885, server_name='0.0.0.0', show_api=False)
