seed=${1:-42}
model_path=${2:-"../../checkpoints/llava-v1.5-7b"}
cd_alpha=${3:-3}
cd_beta=${4:-0.2}
noise_step=${5:-999}
sampling=${9:-"greedy_search"}
mask_mode=${7:-'imccd'}
gamma=${8:-0.2}


declare -A image_folders
image_folders["coco"]="../../../data/coco/val2014"
image_folders["aokvqa"]="../../../data/coco/val2014"
image_folders["gqa"]="../../../data/gqa/images"
types=("random" "popular" "adversarial")

for dataset_name in "${!image_folders[@]}"; do
  image_folder="${image_folders[$dataset_name]}"
  for type in "${types[@]}"; do
    # CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
    # --model-path ${model_path} \
    # --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    # --image-folder ${image_folder} \
    # --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_vcd_alpha1_seed${seed}_${sampling}.jsonl \
    # --cd_alpha $cd_alpha \
    # --cd_beta $cd_beta \
    # --use_cd \
    # --noise_step $noise_step \
    # --seed ${seed} \
    # --sampling ${sampling} \
    # --num_beams 1
    # CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
    # --model-path ${model_path} \
    # --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    # --image-folder ${image_folder} \
    # --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_icd_alpha1_seed${seed}_${sampling}.jsonl \
    # --cd_alpha $cd_alpha \
    # --cd_beta $cd_beta \
    # --use_cd \
    # --use_icd \
    # --noise_step $noise_step \
    # --seed ${seed} \
    # --sampling ${sampling} \
    # --num_beams 1
    # CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
    # --model-path ${model_path} \
    # --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    # --image-folder ${image_folder} \
    # --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}_${sampling}.jsonl \
    # --cd_alpha $cd_alpha \
    # --cd_beta $cd_beta \
    # --noise_step $noise_step \
    # --seed ${seed}
    CUDA_VISIBLE_DEVICES=1 python ./eval/object_hallucination_vqa_llava.py \
    --model-path ${model_path} \
    --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    --image-folder ${image_folder} \
    --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_imccd_seed${seed}_${sampling}.jsonl \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --use_cd \
    --use_mask \
    --mask_mode  imccd \
    --noise_step $noise_step \
    --seed ${seed} \
    --sampling ${sampling} \
    --gamma $gamma 
  done
done

# for dataset_name in "${!image_folders[@]}"; do
#   image_folder="${image_folders[$dataset_name]}"
#   for type in "${types[@]}"; do
#     CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
#     --model-path ${model_path} \
#     --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
#     --image-folder ${image_folder} \
#     --answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_imccd_lowangle_seed${seed}_${sampling}.jsonl \
#     --cd_alpha $cd_alpha \
#     --cd_beta $cd_beta \
#     --use_cd \
#     --use_mask \
#     --mask_mode  imccd \
#     --noise_step $noise_step \
#     --seed ${seed} \
#     --sampling ${sampling} \
#     --gamma $gamma 
#   done
# done





