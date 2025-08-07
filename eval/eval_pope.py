import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gen_files", type=str, default="")

args = parser.parse_args()
print(args.gen_files)

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gen_files)
yes_answers = 0

# compare answers
for index, line in enumerate(gen_files[:-1]):
    # idx = line["question_id"]
    gt_answer = line["label"]
    gen_answer = line["output"]
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos) * 100
recall = true_pos / (true_pos + false_neg) * 100
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions * 100
yes_proportion = yes_answers / total_questions * 100
unknown_prop = unknown / total_questions * 100
# report results
print('Precision: %.2f'%precision)
print('Recall: %.2f'%recall)
print('F1: %.2f'%f1)
print('Accuracy: %.2f'%accuracy)
print('yes: %.2f'%yes_proportion)
print('unknow: %.2f'%unknown_prop)