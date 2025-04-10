import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_json_files(outputs_dir):
    json_files = glob.glob(os.path.join(outputs_dir, "*", "results_*.json"))
    return json_files

def extract_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Get shot count from any task config
    configs = data.get("configs", {})
    if configs:
        first_task = next(iter(configs.values()))
        num_fewshot = first_task.get("num_fewshot", 0)
    else:
        num_fewshot = 0
    # Extract scores
    results = data.get("results", {})
    scores = {}
    for task, metrics in results.items():
        acc = metrics.get("acc,none", None)
        if acc is not None:
            scores[task] = acc
    return num_fewshot, scores

def main():
    outputs_dir = "outputs"
    output_dir = os.path.join(outputs_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    json_files = find_json_files(outputs_dir)

    records = []
    for json_path in json_files:
        model_name = os.path.basename(os.path.dirname(json_path))
        num_fewshot, scores = extract_results(json_path)
        for task, acc in scores.items():
            records.append({
                "model": model_name,
                "task": task,
                "num_fewshot": num_fewshot,
                "accuracy": acc
            })

    df = pd.DataFrame(records)

    for shot in [0, 5]:
        df_shot = df[df["num_fewshot"] == shot]
        if df_shot.empty:
            continue
        plt.figure(figsize=(12, max(6, 0.5 * df_shot['task'].nunique())))
        sns.barplot(
            data=df_shot,
            y="task",
            x="accuracy",
            hue="model",
            orient="h"
        )
        plt.title(f"Model Comparison - {shot}-shot")
        plt.xlabel("Accuracy")
        plt.ylabel("Dataset/Task")
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"model_comparison_{shot}shot.png"))
        plt.close()

    # Compute average accuracy per model and shot count
    avg_df = (
        df.groupby(["model", "num_fewshot"])["accuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    palette = {0: "green", 5: "red"}
    sns.barplot(
        data=avg_df,
        y="model",
        x="accuracy",
        hue="num_fewshot",
        orient="h",
        palette=palette
    )
    plt.title("Average Accuracy per Model (0-shot vs 5-shot)")
    plt.xlabel("Average Accuracy")
    plt.ylabel("Model")
    plt.legend(title="Shots")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_avg.png"))
    plt.close()

if __name__ == "__main__":
    main()