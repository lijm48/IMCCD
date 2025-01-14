dataset_name="mme"

model="llava15"
# model="instructblip"

seed=55


python ./eval/eval_mme.py \
--results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_baseline_seed${seed}/mme_answers



python ./eval/eval_mme.py \
--results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_vcd_seed${seed}/mme_answers


# python ./eval/eval_mme.py \
# --results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_icd_seed${seed}/mme_answers



python ./eval/eval_mme.py \
--results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_imccd_seed${seed}/mme_answers

