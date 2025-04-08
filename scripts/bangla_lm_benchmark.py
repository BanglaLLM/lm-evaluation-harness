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
    parser.add_argument("--models", type=str, help="one or multiple huggingface models, separated by comma")
    parser.add_argument("--tasks", type=str, default=None, help="one or multiple task separated by comma")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--acc_norm", type=bool, default=True)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def eval_models(args, models, tasks):
    results = {}
    
    for model in models:
        start_time = time.time()
        logger.info(f"Evaluating model: {model}")
        model_name = Path(model).name
        eval_output_path = os.path.join(args.output_path, f"{int(start_time)}-{model_name}.json")

        command = (
                f"lm_eval --model hf --model_args pretrained={model} --tasks {','.join(tasks)} "
                f"--num_fewshot {args.num_fewshot}{'' if args.limit is None else f' --limit {args.limit}'} "
                f"--batch_size {args.batch_size} --output_path {eval_output_path} "
                f"--device {args.device}"
            )
        
        logger.info(f"Tasks: {tasks}")
        ###########
        # Using VLM
        ###########
        # max_model_len = 32768 if "qwen" in model.lower() else 65536
        # import torch
        # num_of_gpus = torch.cuda.device_count()
        # command = f"""\
        #     lm_eval --model vllm \
        #         --model_args pretrained={model},tensor_parallel_size={num_of_gpus},dtype=bfloat16,gpu_memory_utilization=0.5,max_model_len={max_model_len},enforce_eager=True \
        #         --tasks {','.join(tasks)} \
        #         --num_fewshot {args.num_fewshot} \
        #         {'' if args.limit is None else f' --limit {args.limit}'} \
        #         --output_path {args.output_path} \
        #         --batch_size {args.batch_size} \
        #         --wandb_args project=s1-bengali,name=eval-{model}-{args.num_fewshot}
        # """

        ##########
        # Using HF
        ##########
        command = f"""\
            lm_eval --model hf \
                --model_args pretrained={model},dtype=bfloat16 \
                --tasks {','.join(tasks)} \
                --num_fewshot {args.num_fewshot} \
                {'' if args.limit is None else f' --limit {args.limit}'} \
                --output_path {args.output_path} \
                --batch_size {args.batch_size} \
                --wandb_args project=s1-bengali,name=eval-{model}-{args.num_fewshot}
        """

        print(
                f"{'=' * 80}\nEvaluating {model} on {', '.join(tasks)}"
            )
        ret = os.system(command)

        # results[model] = (
        #         json.load(open(eval_output_path, encoding="utf-8"))
        #         if ret == 0
        #         else {"results": {}}
        #     )
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

    models = args.models.split(',') if ',' in args.models else [args.models]

    default_tasks = [
        'bangla_mmlu',
        'piqa_bn',
        'openbookqa_bn',
        'commonsenseqa_bn',
        'boolq_bn'
        #'hotpotqa_bn_nllb',
        #'hotpotqa_bn_llama',
        #'bangla_math'
    ]
    tasks = args.tasks.split(',') if args.tasks else default_tasks

    results = eval_models(
        args=args,
        models=models,
        tasks=tasks
    )

    # for task in tasks:
    #     print(
    #         f"|{task} |{'|'.join(map(lambda model: format_value(args, results, model, task), models))}|"
    #     )

if __name__ == "__main__": 
    main()


    

