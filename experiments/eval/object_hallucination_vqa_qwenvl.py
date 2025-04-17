import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

import kornia
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen-vl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        question = '<img>{}</img>{} Answer:'.format(image_path, question)
        questions_id = []
        input_ids = tokenizer([question], return_tensors='pt', padding='longest')

        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = model.transformer.visual.image_transform(image_tensor).unsqueeze(0).to(model.device)

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None   
        
        with torch.inference_mode():
            generate_args = dict(
                input_ids=input_ids.input_ids.cuda(),
                attention_mask=input_ids.attention_mask.cuda(),
                max_new_tokens=20,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                images = image_tensor,
                images_cd=image_tensor_cd,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
                use_kvcache = args.use_kvcache,
                use_mask = args.use_mask,
                mask_mode = args.mask_mode,
                key_pos = key_pos,
                gamma=args.gamma,
            )
            if args.sampling == 'sample':
                generate_args.update(do_sample=True)
            elif args.sampling == 'greedy_search':
                assert args.num_beams == 1, 'greedy search must use num_beams=1!'
                generate_args.update(do_sample=False, num_beams=args.num_beams)
            elif args.sampling == 'beam_search':
                generate_args.update(do_sample=False, num_beams=args.num_beams)
            
            pred = model.generate(**generate_args)

        outputs = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ][0]
        outputs = outputs.strip()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": question,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/ckpt/Qwen-VL")
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

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
<<<<<<< HEAD
=======
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--sampling", type=str, default='sample')
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use_kvcache", action='store_true', default=False)
    
>>>>>>> 11d7567... update
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
