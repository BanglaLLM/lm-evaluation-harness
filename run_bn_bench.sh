python scripts/bangla_lm_benchmark.py \
--models "hishab/mpt-125m-bn-web-book-titulm-tokenizer-hf" \
--batch_size 32 \
--num_fewshot 5 \
--limit 0.5 \
--device "cuda:0" \
--output_path "outputs"
