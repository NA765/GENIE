import os
from diffusers import AutoPipelineForText2Image
import torch
from tqdm import tqdm

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor,  ProcessPoolExecutor
import multiprocessing

SAVED_MODEL_ROOT = "./models" # cached dir for downloaded pretrained weights
NEGATIVE_PROMPTS = "cartoon, unreal, CGI, 3D, fantasy, neon, blurry" # negative prompts for diffusion model generation


# models used for image generation, more models can be found at https://huggingface.co/models?pipeline_tag=text-to-image
model_ids = [
    "sd-legacy/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/sdxl-turbo",
]


# predefined prompt templates
prompts = [
"A serene mountain landscape with a crystal-clear lake reflecting the surrounding peaks at sunrise.",
"A skateboarder grinding along a rail in an urban skatepark, sparks flying from beneath their wheels.",
"A traditional wedding ceremony held outdoors, with flower petals scattered down the aisle.",
# ...
]





def parse_args():
    parser = argparse.ArgumentParser(description='Image generate Pipeline')
    parser.add_argument('--save_image_root', type=str, default='./generated_images', help='image root for saving')
    parser.add_argument('--images_per_cat', type=int, default=6, help='images per category')
    parser.add_argument('--max_workers', type=int, default=4, help='')
    return parser.parse_args()


def load_pipeline(model_id, model_cls=None, device="cuda"):
    cache_dir = os.path.join(SAVED_MODEL_ROOT, model_id)
    os.makedirs(cache_dir, exist_ok=True)
    if model_cls:
        pipe = model_cls.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe



def generate_image(pipe, prompt):
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=50,
            negative_prompts=NEGATIVE_PROMPTS,
        ).images[0]
    except:
        image = pipe(
            prompt=prompt,
            num_inference_steps=50,
            negative_prompt=NEGATIVE_PROMPTS,
        ).images[0]

    return image



def generate_image_and_save(pipe, prompt, save_path):
    if not os.path.exists(save_path):
        image = generate_image(pipe, prompt)
        image.save(save_path)





if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    args = parse_args()

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    for model_id in tqdm(model_ids, desc="Models"):
        # load pipeline
        pipes = [load_pipeline(model_id, device=d) for d in devices]
        save_root = os.path.join(args.save_image_root, os.path.basename(model_id))

        prompt_list = []
        save_path_list = []
        pipe_list = []

        os.makedirs(save_root, exist_ok=True)
        # generate images
        i = 0
        for prompt_id, prompt in (enumerate(prompts)):
            for image_id in range(args.images_per_cat):
                save_path = os.path.join(save_root, f"{prompt_id}_{image_id}.png")
                prompt_list.append(prompt)
                save_path_list.append(save_path)
                pipe_list.append(pipes[i % len(pipes)])
                i += 1


        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(generate_image_and_save, pipe_list, prompt_list, save_path_list),
                      total=len(prompts)*args.images_per_cat, desc="Images"))
        
        # clean pipeline cache
        del pipes
        del prompt_list
        del save_path_list
        del pipe_list
        gc.collect()

        prompt_list = []
        save_path_list = []
        pipe_list = []
