dataset_name="chair"

type="random"
# type="popular"
# type="adversarial"

model="llava15"
# model="instructblip"



coco_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lijiaming/data/coco/annotations/ 
seed=42
sampling="sample"

python ./eval/chair.py \
--cap_file ./output/${dataset_name}/${model}_${dataset_name}_pope_${type}_answers_baseline_seed${seed}_${sampling}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}


python ./eval/chair.py \
--cap_file ./output/${dataset_name}/${model}_${dataset_name}_pope_${type}_answers_vcd_seed${seed}_${sampling}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}



python ./eval/chair.py \
--cap_file ./output/${dataset_name}/${model}_${dataset_name}_pope_${type}_answers_icd_seed${seed}_${sampling}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}

python ./eval/chair.py \
--cap_file ./output/${dataset_name}/${model}_${dataset_name}_pope_${type}_answers_imccd_seed${seed}_${sampling}.jsonl --image_id_key image_id --caption_key caption \
--coco_path ${coco_path}

