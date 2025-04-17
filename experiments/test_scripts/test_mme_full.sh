dataset_name="mme_full"
model="llava15"
# model="instructblip"
seed=42


# python ./eval/eval_mme.py \
# --eval_type full \
# --results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_baseline_seed${seed}/mme_answers 


# python ./eval/eval_mme.py \
# --eval_type full \
# --results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_vcd_seed${seed}/mme_answers


# python ./eval/eval_mme.py \
# --eval_type full \
# --results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_icd_seed${seed}/mme_answers


python ./eval/eval_mme.py \
--eval_type full \
--results_dir ./output/${dataset_name}/${model}_${dataset_name}_answers_imccd_seed42/mme_answers

