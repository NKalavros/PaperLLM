import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd

def fetch_speaker_by_model(log_path='requests_answers.log'):
    # Returns: list of dicts with keys: request_id, model, difficulty, score
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path, 'r') as f:
        for line in f:
            try:
                e = json.loads(line)
                asker = e.get('asker_nickname', '')
                if 'author' in asker.lower():
                    qs = e.get('real_quality_scores') or e.get('quality_scores', {})
                    if set(qs.keys()) == {"Model 1", "Model 2"}:
                        continue
                    if len(qs) < 2:
                        continue
                    difficulty = e.get('question_difficulty', 'All')
                    reqid = e.get('request_id')
                    for m, v in qs.items():
                        if isinstance(v, (int, float)):
                            records.append({
                                'request_id': reqid,
                                'model': m,
                                'difficulty': difficulty,
                                'score': v
                            })
            except Exception:
                continue
    return records

def fetch_llm_by_model(qa_log_path='requests_answers.log'):
    records = []
    reqid_to_diff = {}
    reqid_to_model_order = {}
    if os.path.exists(qa_log_path):
        with open(qa_log_path, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    reqid = e.get('request_id')
                    diff = e.get('question_difficulty', 'All')
                    model_order = None
                    if 'real_rankings' in e and isinstance(e['real_rankings'], dict):
                        model_order = [k for k, v in sorted(e['real_rankings'].items(), key=lambda x: x[1])]
                    elif 'real_quality_scores' in e and isinstance(e['real_quality_scores'], dict):
                        model_order = [k for k, v in sorted(e['real_quality_scores'].items(), key=lambda x: -x[1])]
                    elif 'quality_scores' in e and isinstance(e['quality_scores'], dict):
                        model_order = list(e['quality_scores'].keys())
                    if reqid:
                        reqid_to_diff[reqid] = diff
                        if model_order and len(model_order) == 2:
                            reqid_to_model_order[reqid] = model_order
                except Exception:
                    continue
    import glob
    log_files = glob.glob('judge_results_*.log')
    for log_path in log_files:
        base = os.path.basename(log_path)
        judge_llm = base[len('judge_results_'):-len('.log')]
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    scores = e.get('scores', [])
                    reqid = e.get('request_id')
                    difficulty = reqid_to_diff.get(reqid, 'All')
                    model_order = reqid_to_model_order.get(reqid)
                    if not model_order or len(scores) != 2:
                        continue
                    for idx, answering_llm in enumerate(model_order):
                        answering_llm = answering_llm.lower()
                        score = scores[idx]
                        if answering_llm in ['openai', 'perplexity']:
                            records.append({
                                'difficulty': difficulty,
                                'answering_llm': answering_llm,
                                'judge_llm': judge_llm,
                                'score': score,
                                'request_id': reqid
                            })
                except Exception:
                    continue
    # Also check for legacy judge_results.log (assume judge_llm = 'openai')
    if os.path.exists('judge_results.log'):
        with open('judge_results.log', 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    scores = e.get('scores', [])
                    model_names = e.get('model_names')
                    if not model_names or len(model_names) != 2:
                        continue
                    if len(scores) < 2:
                        continue
                    reqid = e.get('request_id')
                    difficulty = reqid_to_diff.get(reqid, 'All')
                    for idx, answering_llm in enumerate(model_names):
                        answering_llm = answering_llm.lower()
                        score = scores[idx]
                        if answering_llm in ['openai', 'perplexity']:
                            records.append({
                                'difficulty': difficulty,
                                'answering_llm': answering_llm,
                                'judge_llm': 'openai',
                                'score': score,
                                'request_id': reqid
                            })
                except Exception:
                    continue
    return records

def fetch_human_by_model(log_path='requests_answers.log'):
    # Returns: list of dicts with keys: request_id, model, difficulty, score
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path, 'r') as f:
        for line in f:
            try:
                e = json.loads(line)
                asker = e.get('asker_nickname', '')
                if 'author' in asker.lower():
                    continue
                qs = e.get('real_quality_scores') or e.get('quality_scores', {})
                if set(qs.keys()) == {"Model 1", "Model 2"}:
                    continue
                if len(qs) < 2:
                    continue
                difficulty = e.get('question_difficulty', 'All')
                reqid = e.get('request_id')
                for m, v in qs.items():
                    if isinstance(v, (int, float)):
                        records.append({
                            'request_id': reqid,
                            'model': m,
                            'difficulty': difficulty,
                            'score': v
                        })
            except Exception:
                continue
    return records

def plot_scores(llm_data, human_data, speaker_data, outpath="judge_scores.png"):
    # Step 1: Build request_id -> (nickname, Talk) mapping from requests_questions.log
    reqid_to_nickname = {}
    reqid_to_talk = {}
    questions_log = 'requests_questions.log'
    if os.path.exists(questions_log):
        with open(questions_log, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    reqid = e.get('request_id')
                    nickname = e.get('nickname')
                    talk = e.get('speaker')
                    if reqid:
                        reqid_to_nickname[reqid] = nickname
                        reqid_to_talk[reqid] = talk
                except Exception:
                    continue

    allowed_difficulties = ["Easy", "Hard"]
    records = []
    # Audience
    for rec in human_data:
        if rec['difficulty'] not in allowed_difficulties:
            continue
        reqid = rec.get('request_id')
        records.append({
            'difficulty': rec['difficulty'],
            'answering_llm': rec['model'].lower(),
            'judge_llm': 'Audience',
            'score': rec['score'],
            'nickname': reqid_to_nickname.get(reqid),
            'Talk': reqid_to_talk.get(reqid)
        })
    # Speaker
    for rec in speaker_data:
        if rec['difficulty'] not in allowed_difficulties:
            continue
        reqid = rec.get('request_id')
        records.append({
            'difficulty': rec['difficulty'],
            'answering_llm': rec['model'].lower(),
            'judge_llm': 'Speaker',
            'score': rec['score'],
            'nickname': reqid_to_nickname.get(reqid),
            'Talk': reqid_to_talk.get(reqid)
        })
    # LLM Judges
    for rec in llm_data:
        if rec['difficulty'] in allowed_difficulties and rec['answering_llm'] in ['openai', 'perplexity']:
            reqid = rec.get('request_id')
            records.append({
                'difficulty': rec['difficulty'],
                'answering_llm': rec['answering_llm'],
                'judge_llm': rec['judge_llm'],
                'score': rec['score'],
                'nickname': reqid_to_nickname.get(reqid),
                'Talk': reqid_to_talk.get(reqid)
            })
    # Create DataFrame for Easy and Hard
    df_eh = pd.DataFrame(records)
    # Generate 'All' as the union of Easy and Hard
    df_all = df_eh.copy()
    df_all['difficulty'] = 'All'
    # Concatenate
    df = pd.concat([df_eh, df_all], ignore_index=True)
    print("Counts by difficulty:")
    print(df['difficulty'].value_counts())
    df.to_csv('judge_scores_long.csv', index=False)
    print(f"Aggregated data saved to judge_scores_long.csv with {len(df)} rows.")
    import seaborn as sns
    judge_order = ['Audience', 'Speaker'] + sorted([c for c in df['judge_llm'].unique() if c not in ['Audience', 'Speaker']])
    for diff in ["All", "Easy", "Hard"]:
        dfd = df[df["difficulty"] == diff]
        plt.figure(figsize=(max(10, len(judge_order)*1.2), 7))
        ax = sns.boxplot(
            data=dfd,
            x="judge_llm",
            y="score",
            hue="answering_llm",
            order=judge_order,
            palette="Set2",
            showmeans=True
        )
        plt.xlabel("Rater (Judge)")
        plt.ylabel("Score (1–10)")
        plt.title(f"Scores by Rater and Answering LLM ({diff} Difficulty)")
        plt.xticks(rotation=30, ha='right')
        plt.legend(title="Answering LLM")
        plt.tight_layout()
        fname = f"judge_scores_{diff.lower()}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

def main():
    load_dotenv()
    llm_records = fetch_llm_by_model('requests_answers.log')
    speaker_data = fetch_speaker_by_model('requests_answers.log')
    human_data = fetch_human_by_model('requests_answers.log')
    plot_scores(llm_records, human_data, speaker_data)

if __name__ == "__main__":
    main()