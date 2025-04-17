seed=${1:-42}
model_path=${4:-"../../checkpoints/instructblip-vicuna-7b"}
cd_alpha=${3:-3}
cd_beta=${4:-0.2}
noise_step=${5:-999}
sampling=${8:-'sample'}
# sampling=${8:-'greedy_search'}
mask_mode=${9:-'imccd'}

declare -A image_folders
image_folders["coco"]="../../../data/coco/val2014"
image_folders["aokvqa"]="../../../data/coco/val2014"
image_folders["gqa"]="../../../data/gqa/images"
types=("random" "popular" "adversarial")

for dataset_name in "${!image_folders[@]}"; do
  image_folder="${image_folders[$dataset_name]}"
  for type in "${types[@]}"; do
    # CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_instructblip.py \
    # --model-path ${model_path} \
    # --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    # --image-folder ${image_folder} \
    # --answers-file ./output/${dataset_name}/${type}/instructblip_${dataset_name}_pope_${type}_answers_baseline2_seed${seed}_${sampling}.jsonl \
    # --seed ${seed} \
    # --sampling ${sampling} \
    # --num_beams 1
    # CUDA_VISIBLE_DEVICES=1 python ./eval/object_hallucination_vqa_instructblip.py \
    # --model-path ${model_path} \
    # --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    # --image-folder ${image_folder} \
    # --answers-file ./output/${dataset_name}/${type}/instructblip_${dataset_name}_pope_${type}_answers_vcd_alpha1_seed${seed}_${sampling}.jsonl \
    # --cd_alpha $cd_alpha \
    # --cd_beta $cd_beta \
    # --use_cd \
    # --noise_step $noise_step \
    # --seed ${seed} \
    # --sampling ${sampling} \
    # --num_beams 1
    # CUDA_VISIBLE_DEVICES=1 python ./eval/object_hallucination_vqa_instructblip.py \
    # --model-path ${model_path} \
    # --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    # --image-folder ${image_folder} \
    # --answers-file ./output/${dataset_name}/${type}/instructblip_${dataset_name}_pope_${type}_answers_icd_alpha1_seed${seed}_${sampling}.jsonl \
    # --cd_alpha $cd_alpha \
    # --cd_beta $cd_beta \
    # --use_cd \
    # --use_icd \
    # --noise_step $noise_step \
    # --seed ${seed} \
    # --sampling ${sampling} \
    # --num_beams 1
    CUDA_VISIBLE_DEVICES=1 python ./eval/object_hallucination_vqa_instructblip.py \
    --model-path ${model_path} \
    --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    --image-folder ${image_folder} \
    --answers-file ./output/${dataset_name}/${type}/instructblip_${dataset_name}_pope_${type}_answers_imccd_seed${seed}_${sampling}.jsonl \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --use_cd \
    --use_mask \
    --mask_mode  imccd \
    --noise_step $noise_step \
    --seed ${seed} \
    --sampling ${sampling} \
    --num_beams 1
  done
done
