# Bangla MMLU
This dataset is curated from two well-known websites. Both websites contain diverse domains of multiple choice questions(MCQ) of Bangladesh text book and world knowledge.

## Data Instances

An example from an anatomy subtask looks as follows:
```json
{
  "question": "What is the embryological origin of the hyoid bone?",
  "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
  "answer": "D"
}
```

## Data Fields

- `question`: a string feature
- `choices`: a list of 4 string features
- `answer`: a ClassLabel feature

## Data Splits

|       | train   | dev | test |
| ----- | :------:| :-----: | :-----: |
| TOTAL | 304830 | 33871 | 84676