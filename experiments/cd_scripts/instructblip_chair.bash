seed=${1:-55}
dataset_name=${2:-"chair"}
type=${3:-"random"}
model_path=${4:-"../../checkpoints/instructblip-vicuna-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.3}
noise_step=${7:-999}
image_folder=${8:-"../../../data/coco/val2014"}

# python ./eval/chair_eval_instructblip.py \
# --model-path ${model_path} \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/instructblip_${dataset_name}_pope_${type}_answers_without_cd_seed${seed}.jsonl \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --noise_step $noise_step \
# --seed ${seed}


# CUDA_VISIBLE_DEVICES=1  python ./eval/chair_eval_instructblip.py \
# --model-path ${model_path} \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/instructblip_${dataset_name}_pope_${type}_answers_vcd_alpha05_seed${seed}.jsonl \
# --use_cd \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --mask_mode attention \
# --noise_step $noise_step \
# --seed ${seed}


# CUDA_VISIBLE_DEVICES=1  python ./eval/chair_eval_instructblip.py \
# --model-path ${model_path} \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/instructblip_${dataset_name}_pope_${type}_answers_icd_alpha05_seed${seed}.jsonl \
# --use_icd \
# --use_cd \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --mask_mode attention \
# --noise_step $noise_step \
# --seed ${seed}



CUDA_VISIBLE_DEVICES=1 python ./eval/chair_eval_instructblip.py \
--model-path ${model_path} \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}/instructblip_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--use_mask \
--mask_mode imccd \
--noise_step $noise_step \
--seed ${seed}

