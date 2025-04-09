#!/bin/bash

# Create output directory
mkdir -p outputs/comparisons

# Run the comparison script
python scripts/compare_model_results.py \
  --model1_0shot "outputs/Qwen__QWQ-32B/results_2025-04-08T21-01-46.240697.json" \
  --model1_5shot "outputs/Qwen__QWQ-32B/results_2025-04-08T22-53-36.739549.json" \
  --model2_0shot "outputs/BanglaLLM__Bangla-s1k-QWQ-32B-Instruct/results_2025-04-08T20-51-29.632280.json" \
  --model2_5shot "outputs/BanglaLLM__Bangla-s1k-QWQ-32B-Instruct/results_2025-04-08T21-57-48.817161.json" \
  --model1_name "Qwen QWQ-32B" \
  --model2_name "BanglaLLM s1k-QWQ-32B" \
  --output_dir "outputs/comparisons"

echo "Comparison plots generated in outputs/comparisons/"
