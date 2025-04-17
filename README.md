# IMCCD

## 🕹️ Usage
### Environment Setup
Following the VCD, use
```bash
conda create -yn imccd python=3.9
conda activate imccd
cd IMCCD
pip install -r requirements.txt
```

### Evaluation
For evaluation, modify and run the bash in experiments/cd_scripts,
use the 
```bash
cd experiments
sh cd_scripts/llava1.5_pope.bash
```
or directly use the python command:
```bash
cd experiments
CUDA_VISIBLE_DEVICES=0 python ./eval/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/${dataset_name}/${type}/llava15_${dataset_name}_pope_${type}_answers_${mask_mode}_imccd_seed${seed}_${sampling}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--use_cd \
--use_mask \
--mask_mode  $mask_mode \
--noise_step $noise_step \
--seed ${seed} \
--sampling ${sampling}
```

The usage of some important parameters:
  - answers-file: the output answer files
  - use_cd ： use contrastive decoding.
  - cd_alpha： the value of alpha. Valid if and only if use_cd is valid.
  - cd_beta： the value of beta. Valid if and only if use_cd is valid.
  - use_mask: use imccd. Valid if and only if use_cd is valid.
  - mask_mode: change the variants of our method. Valid if and only if use_cd and use_mask are valid. The variants are as follows:
    -  ved .
    -  imccd .
  -  noise_step steps of adding noise in VCD. Valid if and only if use_cd is valid.
  -  sampling: sample or greedy search.

After obtaining the answer files, the evaluation results are estimated by modifying experiments/test_scripts/test_pope.sh or directly using the following code:
```bash
python ./experiments/eval/eval_pope.py \
--gen_files ${answer_file_name} \
--gt_files experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json
```

Similarly, the evalutaion on CHAIR and mme dataset can be conducted by the instructions in experiments/cd_scripts nd experiments/test_scripts.

## Acknowledgment
The code is heavily borrowed from [VCD](https://github.com/DAMO-NLP-SG/VCD/)  and thanks for their contribution.

