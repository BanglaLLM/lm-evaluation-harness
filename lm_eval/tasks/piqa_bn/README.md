# PIQA Bengali (BN)
This is the translation version of [piqa]() LLM evaluation datasets. The datasets were translated using a new method called expressive semantic translation(EST). This method comprises both Google translation and rewrite version using LLM.

## Data details
| Split | Number |
| ----- | ----- |
| Train | 15339 |
| Validation | 1838 |

## Data sample
```json
{
  "goal": "ডিম সেদ্ধ করার পদ্ধতি।",
  "sol1": "ডিমগুলো একটি পাত্রে রাখুন আর এক ইঞ্চি জল দিয়ে ঢাকনা দিন, মাঝারি আঁচে ফুটিয়ে নিন, তারপর ঢাকনা বন্ধ রেখে গ্যাস বন্ধ করে 8 থেকে 10 মিনিট রেখে দিন।",
  "sol2": "ডিমগুলো পাত্রে রাখুন আর সেখানে এক ইঞ্চি ঠাণ্ডা জল দিয়ে ঢাকনা দিন, মাঝারি আঁচে ফুটিয়ে নিন, তারপর ঢাকনা বন্ধ রেখে গ্যাস বন্ধ করে 8 থেকে 10 মিনিট রেখে দিন।",
  "label": 1
}
```