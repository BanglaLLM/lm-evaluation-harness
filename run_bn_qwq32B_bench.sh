limit=1.0

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


python scripts/bangla_lm_benchmark.py \
--models ${models} \
--batch_size "auto:4" \
--num_fewshot 5 \
--device "cuda:0" \
--acc_norm true \
--output_path "outputs"
