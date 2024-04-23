# OpenBookQA Bengali (BN)
This is the translated version of [openbookqa](https://huggingface.co/datasets/allenai/openbookqa) datasets. The datasets were translated using a new method called expressive semantic translation(EST). The method comprises Google machine translation with LLM rewrite-based modification.

## Data details
| Split | Number |
| ---- | ----- |
| train | 4947 |
| validation | 500 |
| test | 497 |

## Data Sample
```json
{
  "question_stem": "রাতে যখন একটি গাড়ি আপনার কাছে আসছে",
  "choices": {
    "text": ["হেডলাইট আরো তীব্র হয়", "হেডলাইট অন্ধকারে ফিরে যায়", "হেডলাইট একটি ধ্রুবক থাকে", "হেডলাইট বন্ধ"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "A"
}
```