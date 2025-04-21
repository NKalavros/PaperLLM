import json
import random
import sys
import uuid
from datetime import datetime

# Predefined list of random nicknames
NICKNAMES = ["alice", "bob", "carol", "dave", "eve", "mallory"]

# Distribution parameters for rankings and quality_scores per model
RANKING_DIST = {
    "Model 1": {"mean": 5, "std": 2},
    "Model 2": {"mean": 5, "std": 2}
}
QUALITY_DIST = {
    "Model 1": {"mean": 7, "std": 2},
    "Model 2": {"mean": 7, "std": 2}
}

def sample_value(dist):
    # Ensure value is an integer and at least 1
    value = max(1, int(round(random.gauss(dist["mean"], dist["std"]))))
    return value

def randomize_entry(entry):
    # Randomize nickname
    entry["nickname"] = random.choice(NICKNAMES)
    # Randomize rankings for each model
    if "rankings" in entry:
        for model in entry["rankings"]:
            entry["rankings"][model] = random.randint(1, 10)
    # Randomize quality_scores for each model
    if "quality_scores" in entry:
        for model in entry["quality_scores"]:
            entry["quality_scores"][model] = random.randint(1, 10)
    return entry

def process_file(input_path, output_path):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if line.strip():
                try:
                    data = json.loads(line)
                    data = randomize_entry(data)
                    outfile.write(json.dumps(data) + "\n")
                except Exception as e:
                    print(f"Error processing line: {e}")

def generate_entry():
    # Generate a new entry with sample distributions
    entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": str(uuid.uuid4()),
        "nickname": random.choice(NICKNAMES),
        "prompt": "sample prompt",  # placeholder
        "file": "sample.pdf",       # placeholder
        "summaries": {},            # placeholder
        "rankings": {},
        "quality_scores": {}
    }
    for model, params in RANKING_DIST.items():
        entry["rankings"][model] = sample_value(params)
    for model, params in QUALITY_DIST.items():
        entry["quality_scores"][model] = sample_value(params)
    return entry

def generate_entries(count, output_path):
    with open(output_path, "w") as outfile:
        for _ in range(count):
            entry = generate_entry()
            outfile.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    # If '--generate' flag is passed, generate new entries; otherwise process file.
    if "--generate" in sys.argv:
        # Optional: allow number of entries to generate via command-line argument
        try:
            count_index = sys.argv.index("--generate") + 1
            count = int(sys.argv[count_index])
        except (IndexError, ValueError):
            count = 5  # default count
        output_file = "/Users/nikolas/Desktop/Projects/PaperSummarizationWeb/generated_entries.log"
        generate_entries(count, output_file)
        print(f"Generated {count} entries written to", output_file)
    else:
        input_file = "/Users/nikolas/Desktop/Projects/PaperSummarizationWeb/requests_answers.log"
        output_file = "/Users/nikolas/Desktop/Projects/PaperSummarizationWeb/randomized_requests_answers.log"
        process_file(input_file, output_file)
        print("Randomized entries written to", output_file)