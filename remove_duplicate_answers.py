import os
import json
import shutil

ANSWERS_LOG = 'requests_answers.log'
QUESTIONS_LOG = 'requests_questions.log'

# Step 1: Find request_ids with duplicate answers
duplicate_ids = set()
with open(ANSWERS_LOG, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
            ma = entry.get('model_answers', {})
            if len(ma) >= 2:
                answers = list(ma.values())
                if answers[0].strip() == answers[1].strip():
                    duplicate_ids.add(entry.get('request_id'))
        except Exception:
            continue

# Step 2: Remove entries with those request_ids from both logs
def filter_log(log_path, ids_to_remove):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    filtered = []
    for line in lines:
        try:
            entry = json.loads(line)
            if entry.get('request_id') not in ids_to_remove:
                filtered.append(line)
        except Exception:
            filtered.append(line)
    # Backup original
    backup = log_path + '.bak'
    if not os.path.exists(backup):
        shutil.copy2(log_path, backup)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered)
    print(f"Filtered {log_path}: removed {len(lines) - len(filtered)} entries.")

filter_log(ANSWERS_LOG, duplicate_ids)
filter_log(QUESTIONS_LOG, duplicate_ids)
print(f"Done. Removed {len(duplicate_ids)} request_ids with duplicate answers.")
