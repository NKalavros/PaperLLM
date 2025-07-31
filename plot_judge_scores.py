# Fetch speaker (author) scores by model from requests_answers.log
def fetch_speaker_by_model(log_path='requests_answers.log'):
    # Returns: data[model][difficulty] = list of scores
    data = defaultdict(lambda: defaultdict(list))
    speaker_ids = set()
    if not os.path.exists(log_path):
        return data, speaker_ids
    with open(log_path, 'r') as f:
        for line in f:
            try:
                e = json.loads(line)
                asker = e.get('asker_nickname', '')
                if 'author' in asker.lower():
                    qs = e.get('real_quality_scores') or e.get('quality_scores', {})
                    model_keys = set(qs.keys())
                    if model_keys == {"Model 1", "Model 2"}:
                        continue  # skip entries with only Model 1/2
                    if len(qs) < 2:
                        continue  # skip entries with only one model
                    difficulty = e.get('question_difficulty', 'All')
                    for m, v in qs.items():
                        if isinstance(v, (int, float)):
                            data[m]['All'].append(v)
                            data[m][difficulty].append(v)
                    speaker_ids.add(e.get('request_id'))
            except json.JSONDecodeError:
                continue
    return data, speaker_ids
import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.stats import ttest_ind

# fetch LLM‐judge scores by model from all judge_results_{model}.log files
def fetch_llm_by_model(qa_log_path='requests_answers.log'):
    # Returns: data[model][difficulty] = list of scores
    data = defaultdict(lambda: defaultdict(list))
    
    # Build request_id -> difficulty map from requests_answers.log
    reqid_to_diff = {}
    if os.path.exists(qa_log_path):
        with open(qa_log_path, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    reqid = e.get('request_id')
                    diff = e.get('question_difficulty', 'All')
                    if reqid:
                        reqid_to_diff[reqid] = diff
                except Exception:
                    continue
    
    # Check for judge results files for each model
    judge_models = ['perplexity', 'gemini', 'claude', 'openai', 'deepseek', 'grok']
    
    for judge_model in judge_models:
        log_path = f'judge_results_{judge_model}.log'
        if not os.path.exists(log_path):
            continue
            
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    scores = e.get('scores', [])
                    model_names = e.get('model_names')
                    if model_names and set(model_names) == {"Model 1", "Model 2"}:
                        continue
                    if len(scores) < 2:
                        continue
                    reqid = e.get('request_id')
                    difficulty = reqid_to_diff.get(reqid, 'All')
                    
                    # For judge results, we typically have two scores for two models
                    # We'll assign them to generic "Model 1" and "Model 2" for the judge
                    data[f'{judge_model}_model1']['All'].append(scores[0])
                    data[f'{judge_model}_model2']['All'].append(scores[1])
                    data[f'{judge_model}_model1'][difficulty].append(scores[0])
                    data[f'{judge_model}_model2'][difficulty].append(scores[1])
                except json.JSONDecodeError:
                    continue
    
    # Also check for legacy judge_results.log
    if os.path.exists('judge_results.log'):
        with open('judge_results.log', 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    scores = e.get('scores', [])
                    model_names = e.get('model_names')
                    if model_names and set(model_names) == {"Model 1", "Model 2"}:
                        continue
                    if len(scores) < 2:
                        continue
                    reqid = e.get('request_id')
                    difficulty = reqid_to_diff.get(reqid, 'All')
                    data['openai_model1']['All'].append(scores[0])
                    data['openai_model2']['All'].append(scores[1])
                    data['openai_model1'][difficulty].append(scores[0])
                    data['openai_model2'][difficulty].append(scores[1])
                except json.JSONDecodeError:
                    continue
                    
    return data

# fetch human scores by model from requests_answers.log
def fetch_human_by_model(log_path='requests_answers.log'):
    # Returns: data[model][difficulty] = list of scores
    data = defaultdict(lambda: defaultdict(list))
    ids = set()
    if not os.path.exists(log_path):
        return data, ids
    with open(log_path, 'r') as f:
        for line in f:
            try:
                e = json.loads(line)
                asker = e.get('asker_nickname', '')
                if 'author' in asker.lower():
                    continue  # skip speaker/author
                qs = e.get('real_quality_scores') or e.get('quality_scores', {})
                model_keys = set(qs.keys())
                if model_keys == {"Model 1", "Model 2"}:
                    continue  # skip entries with only Model 1/2
                if len(qs) < 2:
                    continue  # skip entries with only one model
                difficulty = e.get('question_difficulty', 'All')
                for m, v in qs.items():
                    if isinstance(v, (int, float)):
                        data[m]['All'].append(v)
                        data[m][difficulty].append(v)
                ids.add(e.get('request_id'))
            except json.JSONDecodeError:
                continue
    return data, ids

def plot_scores(llm_data, human_data, speaker_data, outpath="judge_scores.png"):
    # get list of models
    models = sorted(set(llm_data) | set(human_data) | set(speaker_data))
    if not models:
        print("No matching models to plot.")
        return

    judge_types = [
        (llm_data, "LLM Judge", 'lightblue'),
        (human_data, "Audience", 'lightgreen'),
        (speaker_data, "Speaker", 'orange'),
    ]
    difficulties = ["All", "Easy", "Hard"]

    # Prepare data for each (model, judge_type, difficulty)
    box_data = []
    box_labels = []
    box_colors = []
    xtick_positions = []
    pos = 0
    group_width = len(judge_types) * len(difficulties) + 1
    for model in models:
        for judge_idx, (data, judge_label, color) in enumerate(judge_types):
            for diff_idx, diff in enumerate(difficulties):
                scores = data.get(model, {}).get(diff, [])
                box_data.append(scores)
                box_labels.append(f"{model}\n{judge_label}\n{diff}")
                box_colors.append(color)
                xtick_positions.append(pos)
                pos += 1
        pos += 1  # gap between models

    plt.figure(figsize=(max(12, len(box_data) * 0.6), 7))
    bplots = plt.boxplot(box_data, positions=xtick_positions, widths=0.6, patch_artist=True, showmeans=True)
    for patch, color in zip(bplots['boxes'], box_colors):
        patch.set_facecolor(color)

    # Set x-ticks at the center of each model group
    model_centers = []
    for i in range(len(models)):
        start = i * group_width
        end = start + len(judge_types) * len(difficulties)
        model_centers.append((start + end - 1) / 2)
    plt.xticks(model_centers, [m.capitalize() for m in models], fontsize=12)

    # Add legend for judge types and difficulties
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=color, label=label) for _, label, color in judge_types]
    plt.legend(handles=legend_patches, title="Judge Type", loc='upper right')
    # Add difficulty labels below x-axis
    for i, (x, label) in enumerate(zip(xtick_positions, box_labels)):
        plt.text(x, plt.ylim()[0] - 0.5, label.split('\n')[-1], ha='center', va='top', fontsize=9, rotation=90)

    plt.ylabel("Score (1–10)")
    plt.title("LLM Judge vs Audience vs Speaker Scores by Model and Difficulty")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(outpath)
    print(f"Plot saved to {outpath}")
    print(f"Plot saved to {outpath}")

def main():
    load_dotenv()
    llm_data   = fetch_llm_by_model('requests_answers.log')
    speaker_data, speaker_ids = fetch_speaker_by_model('requests_answers.log')
    human_data, human_ids = fetch_human_by_model('requests_answers.log')
    plot_scores(llm_data, human_data, speaker_data)

if __name__ == "__main__":
    main()