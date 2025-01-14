import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
from torchvision import transforms
evolve_vcd_sampling()


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # image_processor.do_normalize = False
    # mean = (0.48145466, 0.4578275, 0.40821073)
    # std = (0.26862954, 0.26130258, 0.27577711)
    # norm = transforms.Normalize(mean, std)

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
    # for img_id in tqdm(range(len(img_files))):
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
        image = Image.open(image_path)    
        # import pdb;pdb.set_trace()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #image_tensor = norm(image_tensor)
        if args.use_icd:
            image_tensor_cd = image_tensor
        elif args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None    

        qs = "Please describe this image in detail."
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
        # conv2 = conv_templates['plain'].copy()
        conv = conv_templates[args.conv_mode].copy()
        # conv.system = conv2.system
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # import pdb;pdb.set_trace()

        conv_cd = conv_templates[args.conv_mode].copy()
        conv_cd.system = "You are a confused object detector."
        # import pdb; pdb.set_trace()
        conv_cd.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv_cd.append_message(conv.roles[1], None)
        prompt_cd = conv_cd.get_prompt()
               
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()   
        input_ids_cd = tokenizer_image_token(prompt_cd, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() 
        _, image_token_indices = torch.where(input_ids  == IMAGE_TOKEN_INDEX)
        len_input_ids = input_ids.shape[1]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)       
        # with torch.inference_mode():
        # import pdb;pdb.set_trace()

        key_pos = [{"image_token_start": image_token_indices,  "image_token_end": image_token_indices + 1 - len_input_ids}]
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
            cd_alpha = args.cd_alpha,
            cd_beta = args.cd_beta,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=1024,
            use_cache=True,
            use_mask = args.use_mask,
            key_pos =key_pos,
            mask_mode = args.mask_mode,
            input_ids_cd = input_ids_cd if args.use_icd else None)
        
        
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        # import pdb;pdb.set_trace()
        img_save["caption"] = outputs
        # outputs = outputs.strip()
        ans_file.write(json.dumps(img_save)+ "\n")
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "model_id": model_name,
        #                            "image": image_file,
        #                            "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
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
    parser.add_argument("--use_mask", action='store_true', default=False)
    parser.add_argument("--mask_mode", type=str, default="gradient")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
