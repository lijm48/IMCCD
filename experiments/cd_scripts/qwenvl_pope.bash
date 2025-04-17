seed=${1:-42}
dataset_name=${2:-"gqa"}
type=${3:-"popular"}
model_path=${4:-"../../checkpoints/Qwen-VL"}
cd_alpha=${5:-3}
cd_beta=${6:-0.2}
noise_step=${7:-999}
sampling=${8:-'sample'}
mask_mode=${9:-'imccd'}
gamma=${10:-0.2}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=../../../data/coco/val2014
else
  image_folder=../../../data/gqa/images
fi
export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_qwenvl.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}/${type}/qwenvl_${dataset_name}_pope_${type}_answers_-1_test_${mask_mode}_seed${seed}_alpha${cd_alpha}_beta${cd_beta}_gamma${gamma}_nzstep${noise_step}_${sampling}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--gamma $gamma \
--use_cd \
--use_mask \
--mask_mode $mask_mode \
--noise_step $noise_step \
--seed ${seed} \
--sampling ${sampling} \
--use_kvcache


