"""
Generating benchmark on Bangla evaluation tasks
"""

import os
import json
import time
from pathlib import Path
from argparse import ArgumentParser
from loguru import logger

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--acc_norm", type=bool, default=False)
    # TODO: implement num_fewshot and limit per task, e.g. task1:5,task2:1:100,task3::1000
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None)
    # TODO: implement hf-auto to pick between causal and seq2seq models so we don't need this
    # Use whatever is faster here
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def eval_hf_models(args, models, tasks):
    results = {}
    
    for model in models:
        start_time = time.time()
        logger.info(f"Evaluating model: {model}")
        model_name = Path(model).name
        eval_output_path = os.path.join(args.output_path, f"{model_name}.json")

        command = (
                f"lm_eval --model hf --model_args pretrained={model} --tasks {','.join(tasks)} "
                f"--num_fewshot {args.num_fewshot}{'' if args.limit is None else f' --limit {args.limit}'} "
                f"--batch_size {args.batch_size} --output_path {eval_output_path}"
            )

        print(
                f"{'=' * 80}\nEvaluating {model} on {', '.join(tasks)} {'=' * 80}"
            )
        ret = os.system(command)

        results[model] = (
                json.load(open(eval_output_path, encoding="utf-8"))
                if ret == 0
                else {"results": {}}
            )
        logger.info(f"total time taken: {time.time() - start_time}")
    
    return results

def extract_value(args, results, model, task, err=False):
    if model not in results:
        return 0
    results = results[model]["results"]
    if task not in results:
        return 0
    results = results[task]
    if args.acc_norm and "acc_norm,none" in results:
        return results["acc_norm,none"] if not err else results["acc_norm_stderr,none"]
    if "acc,none" in results:
        return results["acc,none"] if not err else results["acc_stderr,none"]
    if (args.perplexity or "word_perplexity") + ",none" in results:
        return (
            results[(args.perplexity or "word_perplexity") + ",none"] if not err else 0
        )
    return 0


def format_value(args, results, model, task):
    val = 100 * extract_value(args, results, model, task)
    err = 100 * extract_value(args, results, model, task, err=True)
    return f"{val:.2f}{f' ± {err:.2f}' if err != 0 else ''}"


def format_diff(args, results1, results2, model, task):
    val1 = 100 * extract_value(args, results1, model, task)
    val2 = 100 * extract_value(args, results2, model, task)
    diff = val2 - val1
    return f"**+{diff:.2f}**" if diff > 0 else f"{diff:.2f}"



def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    hf_models = [
        'hishab/mpt-125m-bn-web-book-titulm-tokenizer-hf',
        'hishab/mpt-125m-bn-web-bloom-tokenizer-hf'
    ]

    tasks = [
        'bangla_mmlu',
        'piqa_bn',
        'openbookqa_bn',
        'commonsenseqa_bn'
    ]

    results = eval_hf_models(
        args=args,
        models=hf_models,
        tasks=tasks
    )

    for task in tasks:
        print(
            f"|{task} |{'|'.join(map(lambda model: format_value(args, results, model, task), hf_models))}|"
        )

if __name__ == "__main__": 
    main()


    

