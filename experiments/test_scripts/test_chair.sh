dataset_name="chair"

type="random"
# type="popular"
# type="adversarial"

model="llava15"
# model="instructblip"

coco_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lijiaming/data/coco/annotations/
seed=55

python ./experiments/eval/chair.py \
--cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}

python ./experiments/eval/chair.py \
--cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_cd_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}



python ./experiments/eval/chair.py \
--cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_icd_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}

python ./experiments/eval/chair.py \
--cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}

