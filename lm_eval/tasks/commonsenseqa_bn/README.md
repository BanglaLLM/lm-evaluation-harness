# Commonsense QA Bengali(BN)
This is the translated version of [commonsenseqa](https://huggingface.co/datasets/tau/commonsense_qa) datasets. The datasets were translated by a new method called expressive semantic translation(EST). This translation method comprises both Google machine translation and LLM based rewrite of the translation. Validation set are translated using Google.

## Data details
| Split | Number |
| ----- | ------ |
| Train | 9741 |
| Validation | 1221 |

## Sample data format
```json
{
  "question_stem": "অনেক অগ্নি নির্বাপক যন্ত্র থাকার সম্ভাবনা কোথায়?",
  "choices": {
    "text": [
      "গ্যারেজ",
      "হাসপাতাল",
      "নৌকা",
      "গৃহ",
      "পুতুল ঘর"
    ],
    "label": [
      "A",
      "B",
      "C",
      "D",
      "E"
    ]
  },
  "answerKey": "B"
}
```