import os
import json
import argparse
from tqdm import tqdm


def get_image_annotations(coco_data, image_name, cate = 'person'):
    # 打开并读取COCO注释文件


    # 获取图片ID对应的名称
    image_id = None
    # import pdb;pdb.set_trace()
    for image in coco_data['images']:
        if image['file_name'] == image_name:
            image_id = image['id']
            break

    if image_id is None:
        raise ValueError(f"No image found with name {image_name}")

    # 获取类别ID到名称的映射
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # 获取与该图片相关的所有注释
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # 输出每个物体的类别名称和边界框
    results = []
    exist = False
    for ann in annotations:
        category_name = categories[ann['category_id']]
        if category_name == cate:
            exist = True
        bbox = ann['bbox']  # bbox格式为[x, y, width, height]
        results.append({
            'category': category_name,
            'bbox': bbox
        })

    return results, exist


parser = argparse.ArgumentParser()
parser.add_argument("--gt_files", type=str, default="data/POPE/coco/coco_pope_popular.json")
parser.add_argument("--gen_files", type=str, default="answer_files_POPE/llava15_coco_pope_popular_answers_no_cd.jsonl")
args = parser.parse_args()

# annotation_file = '../../data/coco/annotations/instances_val2014.json' 
# with open(annotation_file, 'r') as f:
#     coco_data = json.load(f)
# open ground truth answers
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]

# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0
exist_hallu_true = 0
exist_hallu_false = 0
exist_count = 0
no_exist_count = 0
no_exist_hallu_true = 0
no_exist_hallu_false = 0

# compare answers
for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    image_name = line['image']
    question = line["text"]

    assert idx == gen_files[index]["question_id"]
    gen_answer = gen_files[index]["text"]
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
                          
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions
# report results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'Accuracy: {accuracy}')
print(f'yes: {yes_proportion}')
print( round(accuracy * 100, 2) , "&" , round(precision * 100, 2)  , "&", round(recall* 100, 2), "&", round(f1 * 100, 2))
# print(f'unknow: {unknown_prop}')