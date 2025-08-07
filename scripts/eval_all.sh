{
export CUDA_VISIBLE_DEVICES='0'
model_name='llava'

seed=17
cd_beta=0.1 # For Generative Task
cd_beta=0.25 # For Discriminative Task
max_len=512
use_cd='true'
cross_jsd_th=0.0001

if [[ $model_name == "llava" ]]; then
  model_path="/path/to/models/llava-v1.5-7b"
elif [[ $model_name == "blip" ]]; then
  model_path="/path/to/models/blip"
elif [[ $model_name == "qwen-chat" ]]; then
  model_path="/path/to/models/Qwen-VL-Chat"
elif [[ $model_name == "qwen2-vl" ]]; then
  model_path="/path/to/models/Qwen2-VL-7B-Instruct"
elif [[ $model_name == "qwen25-vl" ]]; then
  model_path="/path/to/models/Qwen2.5-VL-7B-Instruct"
elif [[ $model_name == "llava-next" ]]; then
  model_path="/path/to/models/llama3-llava-next-8b"
else
  model_path=""
fi

image_folder=/path/to/benchmark/images
dataset='samples'
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path $model_path \
--question-file ../data/${dataset}.json \
--image-folder $image_folder \
--answers-file ./results/${model_name}.${dataset}.len_${max_len}.seed${seed}.jsonl \
--use_cd $use_cd \
--cd_beta $cd_beta \
--max_gen_len $max_len \
--cross_jsd_th $cross_jsd_th \
--seed $seed

}