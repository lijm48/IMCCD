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
import base64
import requests
from PIL import Image
from io import BytesIO
import openai
import traceback
import time

evolve_vcd_sampling()

GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''



def get_gpt4_result(prompt, image_path):
    openai.api_key = None
    openai.api_base = None

    try_cnt = 3
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)
    # import pdb;pdb.set_trace()

    while try_cnt:
        try:
            result = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                # user="1",
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                    }
                ]
            )
            break
        except:
            try_cnt -= 1
            traceback.print_exc()
            print(f'try_cnt=={try_cnt}')
            time.sleep(2)


    # aa = result.to_dict()['choices'][0]['message']['content']
    # print(f'aa=={aa}')
    return result.to_dict()['choices'][0]['message']['content']

def eval_model(args):
    # Model
    disable_torch_init()
    # image_processor.do_normalize = False
    # mean = (0.48145466, 0.4578275, 0.40821073)
    # std = (0.26862954, 0.26130258, 0.27577711)
    # norm = transforms.Normalize(mean, std)

    gpt_answer_records = {}
    assistant_answer_records = {}
    avg_hal_score_1 = 0
    avg_hal_score_2 = 0
    avg_det_score_1 = 0
    avg_det_score_2 = 0
    num_count = 0

    # Questions
    answers1 = [json.loads(q) for q in open(os.path.expanduser(args.answers1), "r")]
    answers2 = [json.loads(q) for q in open(os.path.expanduser(args.answers2), "r")]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    # for img_id in tqdm(range(len(img_files))):
    for idx in tqdm(range(0, len(answers1))):
        if idx > 100:
            break
        img_id = answers1[idx]["image_id"]
        img_id2 = answers2[idx]["image_id"]
        assert img_id == img_id2
        img_file = "COCO_val2014_000000" + str(img_id).zfill(6)+".jpg"

        # img_anns = set(img_info["anns"])
        image_path = os.path.join(args.image_folder, img_file)
        image = Image.open(image_path)    
        # import pdb;pdb.set_trace()
        model_response_1 =  answers1[idx]['caption']
        model_response_2 =  answers2[idx]['caption']
            


        prompt = GPT_JUDGE_PROMPT.format(model_response_1, model_response_2)

        gpt_answer = get_gpt4_result(prompt, image_path)
        print(gpt_answer)
        gpt_answer_records[str(img_id)] = gpt_answer
        print(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" "))
        print(len(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")))
        try:
            hal_score_1, hal_score_2 = gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")[0:2]
            det_score_1, det_score_2 = gpt_answer.split("Detailedness: ")[-1].split("\n")[0].split(" ")[0:2]
            int(hal_score_1[0])
            int(hal_score_2[0])
            int(det_score_1[0])
            int(det_score_2[0])
        except:
            import pdb;pdb.set_trace()
            continue
        avg_hal_score_1 += int(hal_score_1[0])
        avg_hal_score_2 += int(hal_score_2[0])
        avg_det_score_1 += int(det_score_1[0])
        avg_det_score_2 += int(det_score_2[0])
        num_count += 1
        print("=========================================")

        # # dump metric file
        # with open(answers_file, "w") as f:
        #     json.dump(assistant_answer_records, f)

        # dump metric file
        with open(answers_file, "w") as f:
            json.dump(gpt_answer_records, f)

    avg_hal_score_1 = float(avg_hal_score_1) / num_count
    avg_hal_score_2 = float(avg_hal_score_2) / num_count
    avg_det_score_1 = float(avg_det_score_1) / num_count
    avg_det_score_2 = float(avg_det_score_2) / num_count
    print(f"The avg hal score for Assistant 1 and Assistent 2: {avg_hal_score_1}; {avg_hal_score_2}")
    print(f"The avg det score for Assistant 1 and Assistent 2: {avg_det_score_1}; {avg_det_score_2}")
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers1", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers2", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
