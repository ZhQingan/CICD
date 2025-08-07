### Under Construction...
---

# Cross-Image Contrastive Decoding: Precise, Lossless Suppression of Language Priors in Large Vision-Language Models

## Setup

### Environment

```bash
conda create -n cicd -y python=3.10
conda activate cicd

pip install -r requirements.txt
```

**Note**: The dependencies are refered to LLaVA, InstructBLIP, Qwen-VL-Chat. For LLaVA-Next, Qwen2-VL-Instruct, and Qwen2.5-VL-Instruct, you can also easily set up the environment by following the instructions from their official repositories.

### Datasets

All benchmarks need to be processed into structurally consistent JSON files.

Some samples could be found in `data/samples.json`.

You can use `preprocess.py` to build data:
```bash
python preprocess.py \
--image_file /path/to/images \
--data_file /path/to/data.json \
--output_file /path \
--retrieve False
```


## Evaluate CICD


### Quick start

We developed a shell script `scripts/eval_all.sh` that can execute benchmarks end-to-end.

**Inference**

You can also test your own dataset.
```bash
model_name=model_name
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path /path/to/model \
--question-file /path/to/test.json \
--image-folder /path \
--answers-file /path/to/results.json
```

**Evaluation**

```bash
# POPE
python eval_pope.py --ans_folder path/to/results.json
# CHAIR
python test_chair.py --cap_file ../path/to/results.json --coco_path /path/to/coco
```
AMBER and DetailCaps has its own evaluation method; please refer the official repositories for result evaluation:  [AMBER](https://github.com/junyangwang0410/AMBER), [DetailCaps](https://github.com/foundation-multimodal-models/CAPTURE?tab=readme-ov-file)