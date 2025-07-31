import os
import redis
import json
from celery import Celery
import ssl

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
    print(f"Expanding logs using full questions and answers from Redis dump/celery tasks...")

    # --- Step 1: Parse celery task results from Redis dump file ---
    # We'll use the dump file for offline restoration
    dump_path = os.path.join(BASE_DIR, 'redis_dump_20250722_083650.txt')
    if not os.path.exists(dump_path):
        print(f"ERROR: Redis dump file not found: {dump_path}")
        return

    # Build mapping: request_id -> {full_question, answers_by_model}
    request_map = {}
    with open(dump_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    import re
    key_re = re.compile(r'^Key: (celery-task-meta-[a-f0-9\-]+)')
    json_start_re = re.compile(r'^  Type: string, JSON: \{')
    current_key = None
    in_json = False
    json_lines = []
    for idx, line in enumerate(lines):
        m = key_re.match(line)
        if m:
            current_key = m.group(1)
            in_json = False
            json_lines = []
            continue
        if current_key and not in_json and json_start_re.match(line):
            # Start of JSON block
            in_json = True
            json_lines = [line.strip()[len('Type: string, JSON: '):] if line.strip().startswith('Type: string, JSON: ') else line.strip()]
            continue
        if in_json:
            # Continue collecting JSON lines
            if line.strip() == '}' or line.strip().endswith('}'):  # End of JSON block
                json_lines.append(line.strip())
                json_str = '\n'.join(json_lines)
                try:
                    data = json.loads(json_str)
                except Exception:
                    # Try to join lines without newlines
                    try:
                        data = json.loads(''.join(json_lines))
                    except Exception:
                        data = None
                if data and 'result' in data and isinstance(data['result'], dict):
                    rid = data['result'].get('request_id')
                    model = data['result'].get('model')
                    summary = data['result'].get('summary')
                    nickname = data['result'].get('nickname')
                    args = data.get('args', [])
                    question = None
                    # Extract user question: prefer args[-4] if possible, else fallback to previous logic
                    if isinstance(args, list):
                        str_args = [a.strip() for a in args if isinstance(a, str)]
                        if len(str_args) >= 4:
                            question = str_args[-4]
                        elif len(str_args) > 1:
                            # Prefer last string after the first with a question mark
                            q_lines = [a for a in str_args[1:] if '?' in a]
                            if q_lines:
                                question = q_lines[-1]
                            else:
                                question = str_args[-1]
                        elif str_args:
                            question = str_args[0]
                    if rid:
                        if rid not in request_map:
                            request_map[rid] = {'question': question, 'answers': {}, 'nickname': nickname}
                        # Always accumulate all models/answers for this request_id
                        if model and summary:
                            request_map[rid]['answers'][model] = summary
                in_json = False
                current_key = None
                json_lines = []
            else:
                json_lines.append(line.strip())

    # Debug: print models found for each request_id in the dump
    print("Sample of models/answers found in dump:")
    for i, (rid, v) in enumerate(request_map.items()):
        print(f"  {rid}: models={list(v['answers'].keys())}")
        if i >= 4:
            break

    print(f"Parsed {len(request_map)} request_ids from Redis dump.")

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
    import difflib
    def normalized_edit_distance(a, b):
        # Levenshtein distance normalized by length of b (the full answer)
        import numpy as np
        dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
        for i in range(len(a)+1):
            dp[i][0] = i
        for j in range(len(b)+1):
            dp[0][j] = j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[len(a)][len(b)] / max(1, len(b))

    for line in answer_lines:
        try:
            entry = json.loads(line)
        except Exception:
            updated_answers.append(line)
            continue
        rid = entry.get('request_id')
        if rid and rid in request_map:
            celery_summaries = list(request_map[rid]['answers'].values())
            # Expand model_answers dict if present
            if 'model_answers' in entry and isinstance(entry['model_answers'], dict):
                # Try to use real_rankings or real_quality_scores to determine model order
                model_order = []
                if 'real_rankings' in entry and isinstance(entry['real_rankings'], dict):
                    model_order = [k for k, v in sorted(entry['real_rankings'].items(), key=lambda x: x[1])]
                elif 'real_quality_scores' in entry and isinstance(entry['real_quality_scores'], dict):
                    model_order = [k for k, v in sorted(entry['real_quality_scores'].items(), key=lambda x: -x[1])]
                else:
                    model_order = list(request_map[rid]['answers'].keys())

                # Build a pool of available answers (make a copy)
                answer_pool = list(request_map[rid]['answers'].values())
                used_answers = set()
                # Assign answers to 'Model 1', 'Model 2', ... in the order of model_order
                for idx, model in enumerate(model_order):
                    slot = f"Model {idx+1}"
                    # Try to get the answer for this model
                    model_summary = request_map[rid]['answers'].get(model)
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
            if 'model' in entry and 'summary' in entry:
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
