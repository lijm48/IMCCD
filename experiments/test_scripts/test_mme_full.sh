dataset_name="mme_full"

model="llava15"
seed=55


# python ./experiments/eval/eval_mme.py \
# --eval_type full \
# --results_dir ./experiments/output/${dataset_name}/${model}_${dataset_name}_answers_baseline_seed${seed}/mme_answers 


# python ./experiments/eval/eval_mme.py \
# --eval_type full \
# --results_dir ./experiments/output/${dataset_name}/${model}_${dataset_name}_answers_cd500_beta03_seed${seed}/mme_answers


# python ./experiments/eval/eval_mme.py \
# --eval_type full \
# --results_dir ./experiments/output/${dataset_name}/${model}_${dataset_name}_answers_icd_beta03_seed${seed}/mme_answers


python ./experiments/eval/eval_mme.py \
--eval_type full \
--results_dir ./experiments/output/${dataset_name}/${model}_${dataset_name}_answers_imccd_seed${seed}/mme_answers

