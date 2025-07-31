import os
import json
import shutil
import re

REDIS_DUMP = 'redis_dump_20250722_083650.txt'
ANSWERS_LOG = 'requests_answers.log'

# Step 1: Parse Redis dump and build mapping: request_id -> list of summaries (in order)
request_summaries = {}
with open(REDIS_DUMP, 'r', encoding='utf-8') as f:
    lines = f.readlines()

key_re = re.compile(r'^Key: (celery-task-meta-[a-f0-9\-]+)')
json_start_re = re.compile(r'^  Type: string, JSON: \{')
current_key = None
in_json = False
json_lines = []
for line in lines:
    m = key_re.match(line)
    if m:
        current_key = m.group(1)
        in_json = False
        json_lines = []
        continue
    if current_key and not in_json and json_start_re.match(line):
        in_json = True
        json_lines = [line.strip()[len('Type: string, JSON: '):] if line.strip().startswith('Type: string, JSON: ') else line.strip()]
        continue
    if in_json:
        if line.strip() == '}' or line.strip().endswith('}'):  # End of JSON block
            json_lines.append(line.strip())
            json_str = '\n'.join(json_lines)
            try:
                data = json.loads(json_str)
            except Exception:
                try:
                    data = json.loads(''.join(json_lines))
                except Exception:
                    data = None
            if data and 'result' in data and isinstance(data['result'], dict):
                rid = data['result'].get('request_id')
                summary = data['result'].get('summary')
                if rid and summary:
                    if rid not in request_summaries:
                        request_summaries[rid] = []
                    request_summaries[rid].append(summary)
            in_json = False
            current_key = None
            json_lines = []
        else:
            json_lines.append(line.strip())

# Step 2: Replace answers in requests_answers.log by order
with open(ANSWERS_LOG, 'r', encoding='utf-8') as f:
    lines = f.readlines()

updated = []
for line in lines:
    try:
        entry = json.loads(line)
        rid = entry.get('request_id')
        ma = entry.get('model_answers', {})
        if rid in request_summaries and len(ma) >= 2:
            summaries = request_summaries[rid]
            keys = list(ma.keys())
            for i, k in enumerate(keys):
                if i < len(summaries):
                    ma[k] = summaries[i]
            entry['model_answers'] = ma
        updated.append(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        updated.append(line)

# Step 3: Backup and write
backup = ANSWERS_LOG + '.bak'
if not os.path.exists(backup):
    shutil.copy2(ANSWERS_LOG, backup)
with open(ANSWERS_LOG, 'w', encoding='utf-8') as f:
    f.writelines(updated)
print(f"Updated {ANSWERS_LOG} with answers replaced by order from Redis dump.")
