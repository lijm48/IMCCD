seed=${1:-55}
dataset_name=${2:-"chair"}
type=${3:-"random"}
model_path=${4:-"../../checkpoints/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.3}
noise_step=${7:-999}
image_folder=${8:-"../../../data/coco/val2014"}
# if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
#   image_folder=../../../data/coco/val2014
# else
#   image_folder=./data/gqa/images
# fi

# python ./eval/chair_eval.py \
# --model-path ${model_path} \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}.jsonl \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --noise_step $noise_step \
# --seed ${seed}


# CUDA_VISIBLE_DEVICES=1  python ./eval/chair_eval.py \
# --model-path ${model_path} \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_vcd_seed${seed}.jsonl \
# --use_cd \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --mask_mode attention \
# --noise_step $noise_step \
# --seed ${seed}


# CUDA_VISIBLE_DEVICES=0  python ./eval/chair_eval.py \
# --model-path ${model_path} \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_icd_seed${seed}.jsonl \
# --use_cd \
# --use_icd \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --mask_mode attention \
# --noise_step $noise_step \
# --seed ${seed}




CUDA_VISIBLE_DEVICES=1 python ./eval/chair_eval.py \
--model-path ${model_path} \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--use_mask \
--mask_mode imccd \
--noise_step $noise_step \
--seed ${seed}

