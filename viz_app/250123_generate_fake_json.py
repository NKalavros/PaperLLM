import json
import uuid
from datetime import datetime, timedelta
from faker import Faker
import random

fake = Faker()

def generate_random_entry():
    # Generate timestamp within 2025
    timestamp = (datetime(2025, 1, 1) + timedelta(
        days=random.randint(0, 364),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )).isoformat()

    request_id = str(uuid.uuid4())
    prompt_prefix = "Summarize the following papers key findings within 5 lines"
    
    authors = ["Yered Pita-Juarez", "Dimitra Karagkouni", "John Doe", "Jane Smith", 
              "Alex Johnson", "Maria Garcia"]
    text_preview = f"""1  \n  \nA single-nucleus and spatial transcriptomic atlas of the COVID-19 liver reveals topological, functional, 
and regenerative organ disruption in patients  \n{random.choice(authors)}1,2,3*, {" ".join(random.sample(authors, 2))}..."""

    # Generate summaries
    num_summaries = random.randint(1, 3)
    summaries = []
    summary_templates = [
        "This study used single-nucleus RNA-seq and spatial transcriptomics to analyze liver tissue from {} COVID-19 decedents, revealing {}.",
        "Key findings include {} and {} in liver cellular composition.",
        "The research identified {} with expression patterns similar to infected {} cells.",
        "Notable observations included {} and {}, suggesting {}.",
        "Despite lack of clinical {}, the study revealed {} through {} analysis."
    ]
    
    for _ in range(num_summaries):
        summary = random.choice(summary_templates).format(
            random.choice(["17", "multiple", "COVID-19"]),
            random.choice(["significant disruptions in liver structure", "SARS-CoV-2 RNA in hepatocytes", "extensive cellular changes"]),
            random.choice(["hepatocellular injury", "fibrogenesis", "vascular expansion"]),
            random.choice(["Kupffer cell proliferation", "erythrocyte progenitor presence"]),
            random.choice(["lung epithelial", "endothelial", "mesenchymal"]),
            random.choice(["clinical liver injury", "acute symptoms"]),
            random.choice(["spatial transcriptomic", "single-nucleus RNA-seq", "histopathological"])
        )
        summaries.append(summary)

    # Generate rankings and quality scores for all 5 models
    models = ["deepseek", "claude", "gemini", "llama3", "perplexity"]
    
    # Create unique rankings 1-5
    ranks = list(range(1, 6))
    random.shuffle(ranks)
    rankings = {models[i]: ranks[i] for i in range(5)}
    
    # Generate quality scores between 4-9 (can have duplicates)
    quality_scores = {model: random.randint(0, 10) for model in models}

    return {
        "timestamp": timestamp,
        "request_id": request_id,
        "prompt_prefix": prompt_prefix,
        "text_preview": text_preview,
        "summaries": summaries,
        "rankings": rankings,
        "quality_scores": quality_scores
    }

# Generate 5 sample entries
random_entries = [generate_random_entry() for _ in range(1000)]

# Print the generated entries
print(json.dumps(random_entries, indent=2))