seed=${1:-55}
dataset_name=${2:-"coco"}
# dataset_name=${2:-"aokvqa"}
# dataset_name=${2:-"gqa"}
# type=${3:-"random"}
type=${3:-"popular"}
# type=${3:-"adversarial"}
model_path=${4:-"../../checkpoints/llava-v1.5-7b"}
cd_alpha=${5:-3}
cd_beta=${6:-0.2}
noise_step=${7:-999}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=../../../data/coco/val2014
else
  image_folder=../../../data/gqa/images
fi




# CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
# --model-path ${model_path} \
# --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}.jsonl \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --noise_step $noise_step \
# --seed ${seed}




# CUDA_VISIBLE_DEVICES=1 python ./eval/object_hallucination_vqa_llava.py \
# --model-path ${model_path} \
# --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_vcd_seed${seed}.jsonl \
# --cd_alpha $cd_alpha \
# --cd_beta $cd_beta \
# --use_cd \
# --use_icd \
# --noise_step $noise_step \
# --seed ${seed}

# type="popular"
# CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
# --model-path ${model_path} \
# --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
# --image-folder ${image_folder} \
# --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_icd_seed${seed}.jsonl \
# --cd_alpha 1 \
# --cd_beta $cd_beta \
# --use_cd \
# --use_icd \
# --noise_step $noise_step \
# --seed ${seed}


CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--use_cd \
--use_mask \
--mask_mode  imccd \
--noise_step $noise_step \
--seed ${seed}

