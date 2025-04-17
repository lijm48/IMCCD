dataset_name="coco"
# dataset_name="aokvqa"
# dataset_name="gqa"

# type="random"
type="popular"
# type="adversarial"

# # model="qwenvl"
model="llava15"
# model="instructblip"

seed=42
sampling="sample"

results_file="${model}_${dataset_name}_pope_${type}_answers_imccd_seed${seed}_${sampling}.jsonl"


echo ${results_file}
python ./eval/eval_pope.py \
--gen_files ./output/${dataset_name}/${type}/${model}_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl \
--gt_files data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json



