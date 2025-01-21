# Do you summarize papers with LLMs? [WIP]

Well, how to they do? I've built a quick evaluation platform with rankings and scores to gather data from.

Just upload the PDF and it serves answers which you can rate.

## Current supported models (I should really add icons here):

~~GPT4o~~ Damn you rate limits.

Deepseek

Claude

Gemini

Perplexity

More to come

## Installation

### Clone the repo:

```https://github.com/NKalavros/PaperLLM```

### Install the reqs:

```sudo apt-get install redis```

```pip install -r requirements.txt```

### Create the .env file in the same repository:

```
OPENAI_API_KEY="***"
DEEPSEEK_API_KEY="***"
CLAUDE_API_KEY="***"
GEMINI_API_KEY="***"
PERPLEXITY_API_KEY="***"
```

### Run

For this one, I usually keep 3 separate terminals (It's a dev project, don't come after me).

#### Terminal 1 (DB logging):

```
redis-server
```

#### Terminal 2 (Task management):

```
celery -A app.celery worker --loglevel=info --without-heartbeat --without-mingle
```

#### Terminal 3 (Actual app):

```
python app.py
```
