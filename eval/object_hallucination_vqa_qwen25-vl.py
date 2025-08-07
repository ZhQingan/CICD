import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

# import kornia
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM,AutoProcessor
from qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info



tpn_map={
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}
def eval_model(args):
    # Model
    # disable_torch_init()
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen25-vl'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation='eager', torch_dtype=tpn_map[args.torch_type], device_map="cuda",
        # model_path, torch_dtype=torch.float16, device_map="cuda",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        image_file = line["image"]
        question = line["question"]

        image_path = os.path.join(args.image_folder, image_file)

        messages_batch = []
        messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "file://"+image_path},
                    {"type": "text",  "text": question},
                ],
            },
        ]
        messages_batch.append(messages)

        if args.use_cd:
            image_file = line["another_image"]
            another_pt = os.path.join(args.image_folder, image_file)
            if not os.path.exists(another_pt):
                print(f"There is no image: {another_pt}")
                exit()
            messages_cd = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file://"+another_pt},
                        {"type": "text",  "text": question},
                    ],}]
            messages_batch.append(messages_cd)

        text = processor.apply_chat_template(
            messages_batch, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            do_sample=True,
            max_new_tokens=args.max_gen_len,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=False,
            use_cache=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            cd_beta = args.cd_beta,
            cd_alpha = args.cd_alpha,
            cross_jsd_th=args.cross_jsd_th,

        )
        model.img_len=-1
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = outputs[0].strip()
        ans_file.write(json.dumps({"question_id": line['question_id'],
                                   "question": line['question'],
                                   "output": outputs,
                                   "label": line['label'],
                                   "prompt": text[0].replace('<|image_pad|>', ''),
                                #    "prompt": text,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "image_id": line['image_id'],
                                   }) + "\n")
        ans_file.flush()

    ans_file.write(json.dumps(vars(args)) + '\n')
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/ckpt/Qwen-VL")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--max_gen_len", type=int, default=512)
    # parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--use_cd", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--cross_jsd_th", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_type", type=str, default="fp16", choices=['fp16','fp32','bf16'])
    
    args = parser.parse_args()
    set_seed(args.seed)
    if args.use_cd:
        from vcd_utils.vcd_sample_qwen2 import evolve_vcd_sampling
        evolve_vcd_sampling()
    
    print(str(args), flush=True)
    eval_model(args)
