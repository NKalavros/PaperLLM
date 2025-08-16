import os
import re
import json
import ssl
from typing import Dict, Any

import redis
from celery import Celery

# Redis connection (match app.py)
REDIS_URL = 'rediss://red-d0mvcmd6ubrc73epattg:pyQbOXLbZn7yNczcJhQ9MCHYoeKR4045@ohio-keyvalue.render.com:6379'
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
BASE_DIR = os.getcwd()

# Initialize Celery to access task results (match app.py config)
# Add SSL configuration for rediss:// URLs
celery = Celery(
    'app',
    broker=REDIS_URL + '?ssl_cert_reqs=CERT_NONE',
    backend=REDIS_URL + '?ssl_cert_reqs=CERT_NONE'
)
celery.conf.update(
    result_extended=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    broker_connection_retry_on_startup=True,
    # Additional SSL configuration
    broker_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE}
)

def restore_logs_from_redis():
    print("Expanding logs using full questions and answers from Redis dump/celery tasks...")

    # --- Step 0: Find newest redis dump file automatically ---
    dump_candidates = [
        f for f in os.listdir(BASE_DIR)
        if f.startswith('redis_dump_') and f.endswith('.txt') and os.path.isfile(os.path.join(BASE_DIR, f))
    ]
    if not dump_candidates:
        print("ERROR: No redis_dump_*.txt files found in the project root.")
        return
    dump_candidates.sort(reverse=True)
    dump_path = os.path.join(BASE_DIR, dump_candidates[0])
    print(f"Using dump file: {dump_path}")

    # --- Step 1: Parse celery task results and fallback entries from Redis dump file ---
    with open(dump_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Build mapping: request_id -> {question (from celery args or pending_questions.prompt_prefix), answers_by_model, nickname}
    request_map: Dict[str, Dict[str, Any]] = {}

    # 1a) Parse celery-task-meta-* blocks (string JSON)
    key_re = re.compile(r'^Key: (celery-task-meta-[a-f0-9\-]+)')
    json_start_re = re.compile(r'^  Type: string, JSON: \{')
    current_key = None
    in_json = False
    json_lines: list[str] = []
    for line in lines:
        m = key_re.match(line)
        if m:
            current_key = m.group(1)
            in_json = False
            json_lines = []
            continue
        if current_key and not in_json and json_start_re.match(line):
            # Start of JSON block
            in_json = True
            stripped = line.strip()
            prefix = 'Type: string, JSON: '
            json_lines = [stripped[len(prefix):] if stripped.startswith(prefix) else stripped]
            continue
        if in_json:
            json_lines.append(line.strip())
            # Heuristic end: a line of dashes indicates end of section in our dumps
            if line.strip().endswith('}'):
                # Try to parse JSON
                json_str_full = '\n'.join(json_lines)
                data = None
                for candidate in (json_str_full, ''.join(json_lines)):
                    try:
                        data = json.loads(candidate)
                        break
                    except Exception:
                        data = None
                if data and isinstance(data, dict) and 'result' in data and isinstance(data['result'], dict):
                    rid = data['result'].get('request_id')
                    model = data['result'].get('model')
                    summary = data['result'].get('summary')
                    nickname = data['result'].get('nickname')
                    args = data.get('args', [])
                    question = None
                    if isinstance(args, list):
                        str_args = [a.strip() for a in args if isinstance(a, str)]
                        if len(str_args) >= 4:
                            question = str_args[-4]
                        elif len(str_args) > 1:
                            q_lines = [a for a in str_args[1:] if '?' in a]
                            question = q_lines[-1] if q_lines else str_args[-1]
                        elif str_args:
                            question = str_args[0]
                    if rid:
                        if rid not in request_map:
                            request_map[rid] = {'question': question, 'answers': {}, 'nickname': nickname}
                        if model and summary:
                            request_map[rid]['answers'][model] = summary
                # reset
                in_json = False
                current_key = None
                json_lines = []

    # 1b) Parse pending_questions list from FULL LIST CONTENTS
    pq_key_re = re.compile(r'^Key: pending_questions\s*$')
    full_list_marker = re.compile(r'^\s*FULL LIST CONTENTS:')
    item_re = re.compile(r'^\s*\[\d+\]\s+(\{.*\})\s*$')
    in_pq = False
    in_full = False
    for line in lines:
        if pq_key_re.match(line):
            in_pq = True
            in_full = False
            continue
        if in_pq and full_list_marker.match(line):
            in_full = True
            continue
        if in_pq and in_full:
            if line.strip().startswith('['):
                m = item_re.match(line)
                if m:
                    try:
                        obj = json.loads(m.group(1))
                        rid = obj.get('request_id')
                        if rid:
                            question = obj.get('prompt_prefix')
                            nickname = obj.get('nickname')
                            if rid not in request_map:
                                request_map[rid] = {'question': question, 'answers': {}, 'nickname': nickname}
                            else:
                                # Only fill missing fields
                                if not request_map[rid].get('question') and question:
                                    request_map[rid]['question'] = question
                                if not request_map[rid].get('nickname') and nickname:
                                    request_map[rid]['nickname'] = nickname
                    except Exception:
                        pass
                continue
            # Stop when the section delimiter appears
            if line.strip().startswith('-') or line.strip().startswith('Key: '):
                in_pq = False
                in_full = False

    # Debug: small sample
    print("Sample request_ids parsed:")
    for i, (rid, v) in enumerate(request_map.items()):
        print(f"  {rid}: question?={'yes' if bool(v.get('question')) else 'no'}, answers={list(v.get('answers', {}).keys())}")
        if i >= 4:
            break
    print(f"Parsed {len(request_map)} request_ids from dump.")

    # --- Step 2: Expand requests_questions.log ---
    questions_log = os.path.join(BASE_DIR, 'requests_questions.log')
    # Ensure file exists
    if not os.path.exists(questions_log):
        open(questions_log, 'a', encoding='utf-8').close()
    with open(questions_log, 'r', encoding='utf-8') as fq:
        question_lines = fq.readlines()
    updated_questions = []
    for line in question_lines:
        try:
            entry = json.loads(line)
        except Exception:
            updated_questions.append(line)
            continue
        rid = entry.get('request_id')
        if rid and rid in request_map and request_map[rid].get('question'):
            # Extract the last non-empty line ending with a question mark, else last non-empty line
            full_q = request_map[rid]['question']
            user_q = None
            if isinstance(full_q, str):
                lines = [l.strip() for l in full_q.strip().split('\n') if l.strip()]
                # Prefer last line ending with a question mark
                q_lines = [l for l in lines if l.endswith('?')]
                if q_lines:
                    user_q = q_lines[-1]
                elif lines:
                    user_q = lines[-1]
                else:
                    user_q = full_q.strip()
            else:
                user_q = full_q
            entry['prompt'] = user_q
        updated_questions.append(json.dumps(entry, ensure_ascii=False) + '\n')
    # Backup and overwrite
    backup_q = questions_log + '.bak'
    if not os.path.exists(backup_q):
        import shutil
        shutil.copy2(questions_log, backup_q)
        print(f"Created backup: {backup_q}")
    with open(questions_log, 'w', encoding='utf-8') as fq:
        fq.writelines(updated_questions)
    print(f"Expanded questions in requests_questions.log.")

    # --- Step 3: Expand requests_answers.log ---
    answers_log = os.path.join(BASE_DIR, 'requests_answers.log')
    # Ensure file exists
    if not os.path.exists(answers_log):
        open(answers_log, 'a', encoding='utf-8').close()
    with open(answers_log, 'r', encoding='utf-8') as fa:
        answer_lines = fa.readlines()
    updated_answers = []
    def normalized_edit_distance(a: str, b: str) -> float:
        # Pure Python Levenshtein distance normalized by length of b (the full answer)
        la, lb = len(a), len(b)
        if lb == 0:
            return 0.0 if la == 0 else 1.0
        # Initialize DP rows to save memory
        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)
        for i in range(1, la + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev, curr = curr, prev
        return prev[lb] / lb

    for line in answer_lines:
        try:
            entry = json.loads(line)
        except Exception:
            updated_answers.append(line)
            continue
        rid = entry.get('request_id')
        if rid and rid in request_map:
            celery_summaries = list(request_map[rid].get('answers', {}).values())
            # Expand model_answers dict if present
            if 'model_answers' in entry and isinstance(entry['model_answers'], dict):
                # Try to use real_rankings or real_quality_scores to determine model order
                model_order = []
                if 'real_rankings' in entry and isinstance(entry['real_rankings'], dict):
                    model_order = [k for k, v in sorted(entry['real_rankings'].items(), key=lambda x: x[1])]
                elif 'real_quality_scores' in entry and isinstance(entry['real_quality_scores'], dict):
                    model_order = [k for k, v in sorted(entry['real_quality_scores'].items(), key=lambda x: -x[1])]
                else:
                    model_order = list(request_map[rid].get('answers', {}).keys())

                # Build a pool of available answers (make a copy)
                answer_pool = list(request_map[rid].get('answers', {}).values())
                used_answers = set()
                # Assign answers to 'Model 1', 'Model 2', ... in the order of model_order
                for idx, model in enumerate(model_order):
                    slot = f"Model {idx+1}"
                    # Try to get the answer for this model
                    model_summary = request_map[rid].get('answers', {}).get(model)
                    chosen = None
                    # If model_summary is in the pool and not used, use it
                    if model_summary and model_summary in answer_pool and model_summary not in used_answers:
                        chosen = model_summary
                    else:
                        # Otherwise, pick the first unused answer from the pool
                        for ans in answer_pool:
                            if ans not in used_answers:
                                chosen = ans
                                break
                    if chosen:
                        entry['model_answers'][slot] = chosen
                        used_answers.add(chosen)
            # Expand summary fields if present
            if 'model' in entry and 'summary' in entry and celery_summaries:
                ans = entry['summary']
                best_full = None
                best_dist = 1.0
                for full in celery_summaries:
                    if not full:
                        continue
                    if ans and (full.startswith(ans) or ans in full):
                        best_full = full
                        best_dist = 0
                        break
                    dist = normalized_edit_distance(ans, full)
                    if dist < best_dist:
                        best_dist = dist
                        best_full = full
                if best_full and (best_dist < 0.25 or (ans and best_full.startswith(ans))):
                    entry['summary'] = best_full
        updated_answers.append(json.dumps(entry, ensure_ascii=False) + '\n')
    # Backup and overwrite
    backup_a = answers_log + '.bak'
    if not os.path.exists(backup_a):
        import shutil
        shutil.copy2(answers_log, backup_a)
        print(f"Created backup: {backup_a}")
    with open(answers_log, 'w', encoding='utf-8') as fa:
        fa.writelines(updated_answers)
    print(f"Expanded answers in requests_answers.log.")

    print("Restoration and expansion complete.")

if __name__ == "__main__":
    restore_logs_from_redis()
