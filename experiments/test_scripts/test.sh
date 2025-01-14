dataset_name="coco"
# dataset_name="aokvqa"
# dataset_name="gqa"

# type="random"
type="popular"
# type="adversarial"

model="llava15"
# model="instructblip"

seed=55

# python ./eval/eval_pope.py \
# --gen_files ./output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}.jsonl \
# --gt_files data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json

# python ./eval/eval_pope.py \
# --gen_files ./output/${dataset_name}/${model}_${dataset_name}_pope_${type}_answers_cd_seed${seed}.jsonl \
# --gt_files data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json

# python ./eval/eval_pope.py \
# --gen_files ./output/${dataset_name}/${model}_${dataset_name}_pope_${type}_answers_icd_seed${seed}.jsonl \
# --gt_files data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json


echo ${model}_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl 
python ./eval/eval_pope.py \
--gen_files ./output/${dataset_name}/${type}/${model}_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl \
--gt_files data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json
