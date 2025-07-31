def jaccard_similarity(a, b):
    # Tokenize by whitespace, lowercase
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
import os
import json
import argparse
import numpy as np
from scipy.stats import ttest_rel
from dotenv import load_dotenv
from openai import OpenAI
import requests  # added for Claude
import glob
import PyPDF2

# Prefix and suffix for the judge prompt
PREFIX = (
    "You are an LLM judge who will be presented with a talk, a question and 2 answers. "
    "Return ONLY two integers from 1 to 10 corresponding to the quality of model 1 and model 2 "
    "answers to the question respectively. Respond with numbers only, no additional text. Be very strict in your assessment."
)
SUFFIX = "Provide only two integers separated by a space."


# Prefix and suffix for the judge prompt
PREFIX = (
    "You are an LLM judge who will be presented with a talk, a question and 2 answers. "
    "Return ONLY two integers from 1 to 10 corresponding to the quality of model 1 and model 2 "
    "answers to the question respectively. Respond with numbers only, no additional text. Be very strict in your assessment."
)
SUFFIX = "Provide only two integers separated by a space."

def build_prompt(talk, question, ans1, ans2):
    return f"{PREFIX}\nTalk: {talk}\nQuestion: {question}\nAnswer1: {ans1}\nAnswer2: {ans2}\n{SUFFIX}"

def load_questions(log_path):
    questions = {}
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'prompt' in entry and 'request_id' in entry and 'file' in entry:
                questions[entry['request_id']] = {
                    'talk': entry['file'],
                    'question': entry['prompt']
                }
    return questions

def load_answers(log_path):
    answers = {}
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'request_id' in entry and 'model_answers' in entry:
                answers[entry['request_id']] = entry['model_answers']
    return answers

def judge_with_openai(client, prompt, model_name):
    import time
    delays = [30, 60, 90, 120, 150, 180]
    for attempt, delay in enumerate(delays + [None]):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            content = json.loads(resp.json())['choices'][0]['message']['content'].strip()
            nums = [int(x) for x in content.split() if x.isdigit()]
            return nums[:2]
        except Exception as e:
            msg = str(e)
            # Check for 529, 429, or rate limit
            if '529' in msg or '429' in msg or 'rate limit' in msg.lower() or 'too many requests' in msg.lower():
                if delay is not None:
                    print(f"OpenAI rate limit error, retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"OpenAI rate limit error, all retries failed. Skipping.")
                    return []
            else:
                print(f"OpenAI error: {e}")
                return []

def judge_with_claude(api_key, prompt, model_name):
    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192
    }
    import time
    delays = [30, 60, 90, 120, 150, 180]
    for attempt, delay in enumerate(delays + [None]):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            response_data = resp.json()
            content = response_data['content'][0]['text'].strip()
            print(content)
            nums = [int(x) for x in content.split() if x.isdigit()]
            return nums[:2]
        except Exception as e:
            msg = str(e)
            # Claude 529/429/rate limit handling
            if '529' in msg or '429' in msg or 'rate limit' in msg.lower() or 'too many requests' in msg.lower():
                if delay is not None:
                    print(f"Claude rate limit error, retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Claude rate limit error, all retries failed. Skipping.")
                    return []
            else:
                print(f"Claude error: {e}")
                return []

def judge_with_perplexity(api_key, prompt, model_name):
    """Judge using Perplexity API (OpenAI-compatible)"""
    endpoint = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192
    }
    import time
    delays = [30, 60, 90, 120, 150, 180]
    for attempt, delay in enumerate(delays + [None]):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            response_data = resp.json()
            content = response_data['choices'][0]['message']['content'].strip()
            print(content)
            nums = [int(x) for x in content.split() if x.isdigit()]
            return nums[:2]
        except Exception as e:
            msg = str(e)
            if '529' in msg or '429' in msg or 'rate limit' in msg.lower() or 'too many requests' in msg.lower():
                if delay is not None:
                    print(f"Perplexity rate limit error, retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Perplexity rate limit error, all retries failed. Skipping.")
                    return []
            else:
                print(f"Perplexity error: {e}")
                return []

def judge_with_gemini(model_name, prompt):
    """Judge using Google Gemini API via google-genai client"""
    import time
    import google.genai as genai
    delays = [30, 60, 90, 120, 150, 180]
    client = genai.Client()
    for attempt, delay in enumerate(delays + [None]):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            content = response.text.strip() if response.text is not None else ""
            if not content and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content is not None and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    part = candidate.content.parts[0]
                    if hasattr(part, 'text') and part.text:
                        content = part.text.strip()
            print(content)
            nums = [int(x) for x in content.split() if x.isdigit()]
            return nums[:2]
        except Exception as e:
            msg = str(e)
            if '529' in msg or '429' in msg or 'rate limit' in msg.lower() or 'too many requests' in msg.lower():
                if delay is not None:
                    print(f"Gemini rate limit error, retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Gemini rate limit error, all retries failed. Skipping.")
                    return []
            else:
                print(f"Gemini error: {e}")
                return []

def judge_with_deepseek(api_key, prompt, model_name):
    """Judge using DeepSeek API (OpenAI-compatible)"""
    endpoint = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192
    }
    import time
    delays = [30, 60, 90, 120, 150, 180]
    for attempt, delay in enumerate(delays + [None]):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            response_data = resp.json()
            content = response_data['choices'][0]['message']['content'].strip()
            print(content)
            nums = [int(x) for x in content.split() if x.isdigit()]
            return nums[:2]
        except Exception as e:
            msg = str(e)
            if '529' in msg or '429' in msg or 'rate limit' in msg.lower() or 'too many requests' in msg.lower():
                if delay is not None:
                    print(f"DeepSeek rate limit error, retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"DeepSeek rate limit error, all retries failed. Skipping.")
                    return []
            else:
                print(f"DeepSeek error: {e}")
                return []

def judge_with_grok(api_key, prompt, model_name):
    """Judge using Grok (xAI) API (OpenAI-compatible)"""
    endpoint = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192
    }
    import time
    delays = [30, 60, 90, 120, 150, 180]
    for attempt, delay in enumerate(delays + [None]):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            response_data = resp.json()
            content = response_data['choices'][0]['message']['content'].strip()
            print(content)
            nums = [int(x) for x in content.split() if x.isdigit()]
            return nums[:2]
        except Exception as e:
            msg = str(e)
            if '529' in msg or '429' in msg or 'rate limit' in msg.lower() or 'too many requests' in msg.lower():
                if delay is not None:
                    print(f"Grok rate limit error, retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Grok rate limit error, all retries failed. Skipping.")
                    return []
            else:
                print(f"Grok error: {e}")
                return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['perplexity', 'gemini', 'claude', 'openai', 'deepseek', 'grok'], default='openai')
    parser.add_argument('--model-name', default=None, help='Model name for the selected provider (see docs for options)')
    parser.add_argument('--questions-log', default='requests_questions.log')
    parser.add_argument('--answers-log', default='requests_answers.log')
    parser.add_argument('--test', action='store_true', help='If set, only run one random question for testing.')
    args = parser.parse_args()

    load_dotenv()

    # Set up API clients/keys based on model
    client = None
    api_key = None

    # Default model names for each provider
    default_models = {
        'openai': 'o4-mini',
        'claude': 'claude-opus-4-20250514',
        'perplexity': 'sonar-pro',
        'gemini': 'gemini-2.5-flash',
        'deepseek': 'deepseek-chat',
        'grok': 'grok-4',
    }
    model_name = args.model_name or default_models[args.model]

    if args.model == 'openai':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY2'))
    elif args.model == 'claude':
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY not set")
    elif args.model == 'perplexity':
        api_key = os.getenv('PERPLEXITY_API_KEY2')
        if not api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set")
    elif args.model == 'gemini':
        # google-genai client uses env var GEMINI_API_KEY
        pass
    elif args.model == 'deepseek':
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
    elif args.model == 'grok':
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise RuntimeError("GROK_API_KEY not set")
    else:
        raise RuntimeError(f"Unsupported judge model: {args.model}")

    # Load all Talk*.pdf files at startup
    talk_texts = {}
    for pdf_path in glob.glob('Talk*.pdf'):
        base = os.path.basename(pdf_path)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or '' for page in reader.pages)
            talk_texts[base] = text
        except Exception as e:
            print(f"Failed to load {pdf_path}: {e}")

    questions = load_questions(args.questions_log)
    answers = load_answers(args.answers_log)

    # Map each question's 'file' to the loaded talk text
    for q in questions.values():
        talk_file = os.path.basename(q['talk']) if q['talk'] else None
        if talk_file and talk_file in talk_texts:
            q['talk'] = talk_texts[talk_file]
        else:
            q['talk'] = f"[Talk file {talk_file} not found]"

    import random
    results = {}
    rids = list(questions.keys())
    if args.test:
        # Pick one random question id
        if not rids:
            print("No questions available for testing.")
            return
        rids = [random.choice(rids)]
        print(f"[TEST MODE] Running only for request_id: {rids[0]}")

    for rid in rids:
        q = questions[rid]
        ma = answers.get(rid, {})
        if len(ma) >= 2:
            # Determine model order from real_rankings or real_quality_scores if available
            entry = None
            # Try to find the original entry for this request_id in answers log for real_rankings
            # (Assume answers log lines are available in memory, else skip this step)
            # Fallback: use keys as before
            model_order = []
            # Try to get real_rankings or real_quality_scores from ma if present
            if 'real_rankings' in ma and isinstance(ma['real_rankings'], dict):
                model_order = [k for k, v in sorted(ma['real_rankings'].items(), key=lambda x: x[1])]
            elif 'real_quality_scores' in ma and isinstance(ma['real_quality_scores'], dict):
                model_order = [k for k, v in sorted(ma['real_quality_scores'].items(), key=lambda x: -x[1])]
            else:
                # Try to get from q if present
                if 'real_rankings' in q and isinstance(q['real_rankings'], dict):
                    model_order = [k for k, v in sorted(q['real_rankings'].items(), key=lambda x: x[1])]
                elif 'real_quality_scores' in q and isinstance(q['real_quality_scores'], dict):
                    model_order = [k for k, v in sorted(q['real_quality_scores'].items(), key=lambda x: -x[1])]
                else:
                    # Fallback: use sorted keys
                    model_order = sorted(ma.keys())

            # Map Model 1, Model 2, ... to the correct model names
            ans1 = None
            ans2 = None
            if len(model_order) >= 2:
                ans1 = ma.get(f"Model 1")
                ans2 = ma.get(f"Model 2")
                # If not present, try to get by model name
                if ans1 is None:
                    ans1 = ma.get(model_order[0])
                if ans2 is None:
                    ans2 = ma.get(model_order[1])
            else:
                # Fallback: use sorted keys
                keys = sorted(ma.keys())
                ans1 = ma[keys[0]]
                ans2 = ma[keys[1]]

            # Check similarity before judging
            if ans1 and ans2:
                sim = jaccard_similarity(ans1, ans2)
                if sim >= 0.9:
                    print(f"Skipping request_id {rid}: answers too similar (Jaccard similarity={sim:.2f})")
                    continue
            prompt = build_prompt(q['talk'], q['question'], ans1, ans2)
            # show last 100 chars of each prompt part for review
            parts = {
                'PREFIX': PREFIX[-100:],
                'TALK': q['talk'][-100:],
                'QUESTION': q['question'][-100:],
                'ANSWER1': ans1[-300:] if ans1 else '',
                'ANSWER2': ans2[-300:] if ans2 else '',
                'SUFFIX': SUFFIX[-100:]
            }
            for name, snippet in parts.items():
                print(f"{name} tail (last 100 chars): {snippet}")
            if args.model == 'openai':
                scores = judge_with_openai(client, prompt, model_name)
            elif args.model == 'claude':
                scores = judge_with_claude(api_key, prompt, model_name)
            elif args.model == 'perplexity':
                scores = judge_with_perplexity(api_key, prompt, model_name)
            elif args.model == 'gemini':
                scores = judge_with_gemini(model_name, prompt)
            elif args.model == 'deepseek':
                scores = judge_with_deepseek(api_key, prompt, model_name)
            elif args.model == 'grok':
                scores = judge_with_grok(api_key, prompt, model_name)
            else:
                raise RuntimeError(f"Unsupported judge model: {args.model}")
            if scores and isinstance(scores, list) and len(scores) == 2:
                results[rid] = scores
                print(json.dumps({'request_id': rid, 'scores': scores}))

    # append to a model-specific log
    output_file = f'judge_results_{args.model}.log'
    with open(output_file, 'a') as outf:
        for rid, sc in results.items():
            outf.write(json.dumps({'request_id': rid, 'scores': sc}) + '\n')
    # compute overall stats
    s1 = [sc[0] for sc in results.values()]
    s2 = [sc[1] for sc in results.values()]
    if s1 and s2:
        mean1, mean2 = np.mean(s1), np.mean(s2)
        se1 = np.std(s1, ddof=1)/np.sqrt(len(s1))
        se2 = np.std(s2, ddof=1)/np.sqrt(len(s2))
        p = ttest_rel(s1, s2).pvalue
        stats = {
            'mean_model1': round(mean1,2), 'se_model1': round(se1,2),
            'mean_model2': round(mean2,2), 'se_model2': round(se2,2),
            'p_value': round(p,4)
        }
        print(json.dumps({'summary_stats': stats}))

if __name__ == '__main__':
    main()