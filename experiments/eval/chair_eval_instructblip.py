import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

from transformers import set_seed
from lavis.models import load_model_and_preprocess
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
import random
evolve_vcd_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    img_files = os.listdir(args.image_folder)
    random.shuffle(img_files)
    with open(os.path.expanduser(args.image_folder + '/../annotations/instances_val2014.json'), 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])

    img_dict = {}

    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}
    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )
    # Questions
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for img_id in tqdm(range(500)):
        if img_id == 500:
            break
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[img_id]     
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id
        image_path = os.path.join(args.image_folder, img_file)
        raw_image = Image.open(image_path).convert("RGB") 

        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        ## create a white image for contrastive decoding
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      
        qs = "Please describe this image in detail."
        prompt = qs

        with torch.inference_mode():
            outputs = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=image_tensor_cd, cd_beta = args.cd_beta, cd_alpha = args.cd_alpha,
                use_mask = args.use_mask, mask_mode = args.mask_mode, use_icd = args.use_icd)

        outputs = outputs[0]
        img_save["caption"] = outputs
        # outputs = outputs.strip()
        ans_file.write(json.dumps(img_save)+ "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_icd", action='store_true', default=False)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mask", action='store_true', default=False)
    parser.add_argument("--mask_mode", type=str, default="gradient")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
