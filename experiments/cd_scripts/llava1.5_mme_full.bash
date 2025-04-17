seed=${1:-42}
dataset_name=${2:-"mme_full"}
model=${3:-"llava"}
model_path=${4:-"../../checkpoints/llava-v1.5-7b"}

cd_alpha=${5:-1}
cd_beta=${6:-0.5}
noise_step=${7:-500}
sampling=${8:-'sample'}
mask_mode=${9:-'imccd'}

image_folder=../../../data/MME_Benchmark_release_version/

# CUDA_VISIBLE_DEVICES=0 python ./eval/mme_llava.py \
# --model-path ${model_path} \
# --question-file ../../../data/MME_Benchmark_release_version/mme_full.jsonl \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/llava15_${dataset_name}_answers_baseline_seed${seed}.jsonl \
# --seed ${seed} \
# --sampling ${sampling} 


# CUDA_VISIBLE_DEVICES=0 python ./eval/mme_llava.py \
# --model-path ${model_path} \
# --question-file ../../../data/MME_Benchmark_release_version/mme_full.jsonl \
# --image-folder ${image_folder} \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --use_cd \
# --use_icd \
# --noise_step $noise_step \
# --answers-file ./output/${dataset_name}/llava15_${dataset_name}_answers_icd_seed${seed}.jsonl \
# --seed ${seed} \
# --sampling ${sampling} 

# CUDA_VISIBLE_DEVICES=0 python ./eval/mme_llava.py \
# --model-path ${model_path} \
# --question-file ../../../data/MME_Benchmark_release_version/mme_full.jsonl \
# --image-folder ${image_folder} \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --use_cd \
# --noise_step $noise_step \
# --answers-file ./output/${dataset_name}/llava15_${dataset_name}_answers_vcd_seed${seed}.jsonl \
# --seed ${seed} \
# --sampling ${sampling} 

CUDA_VISIBLE_DEVICES=1 python ./eval/mme_llava.py \
--model-path ${model_path} \
--question-file ../../../data/MME_Benchmark_release_version/mme_full.jsonl \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}/llava15_${dataset_name}_answers_imccd_3lowangle_alpha05_seed${seed}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--use_cd \
--use_mask \
--mask_mode  $mask_mode \
--noise_step $noise_step \
--seed ${seed} \
--sampling ${sampling} 









# python ./eval/convert_answer_to_mme.py \
# --output_path ./output/${dataset_name}/llava15_${dataset_name}_answers_baseline_seed${seed}.jsonl \
# --seed ${seed} \
# --log_path ./output/${dataset_name}/llava15_${dataset_name}_answers_baseline_seed${seed}/ \



# python ./eval/convert_answer_to_mme.py \
# --output_path ./output/${dataset_name}/llava15_${dataset_name}_answers_vcd_seed${seed}.jsonl \
# --seed ${seed} \
# --log_path ./output/${dataset_name}/llava15_${dataset_name}_answers_vcd_seed${seed}/ \

# python ./eval/convert_answer_to_mme.py \
# --output_path ./output/${dataset_name}/llava15_${dataset_name}_answers_icd_seed${seed}.jsonl \
# --seed ${seed} \
# --log_path ./output/${dataset_name}/llava15_${dataset_name}_answers_icd_seed${seed}/ \

python ./eval/convert_answer_to_mme.py \
--output_path ./output/${dataset_name}/llava15_${dataset_name}_answers_imccd_3lowangle_alpha05_seed${seed}.jsonl \
--seed ${seed} \
--log_path ./output/${dataset_name}/llava15_${dataset_name}_answers_imccd_3lowangle_alpha05_seed${seed}/ \