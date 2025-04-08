limit=1.0
models="\
BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct,\
Qwen/Qwen2.5-3B-Instruct,\
BanglaLLM/Bangla-s1k-llama-3.2-3B-Instruct,\
meta-llama/Llama-3.2-3B-Instruct,\
hishab/titulm-llama-3.2-3b-v2.0"

models="\
BanglaLLM/Bangla-s1k-qwen-2.5-32B-Instruct,\
Qwen/Qwen2.5-32B-Instruct"

models="\
BanglaLLM/Bangla-s1k-QWQ-32B-Instruct,\
Qwen/QWQ-32B"

# limit=0.1
# models="BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct"

python scripts/bangla_lm_benchmark.py \
--models ${models} \
--batch_size "auto:4" \
--num_fewshot 0 \
--device "cuda:0" \
--acc_norm true \
--output_path "outputs"


# python scripts/bangla_lm_benchmark.py \
# --models ${models} \
# --batch_size "auto:4" \
# --num_fewshot 5 \
# --device "cuda:0" \
# --acc_norm true \
# --output_path "outputs"
