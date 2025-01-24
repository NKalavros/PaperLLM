import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template #type: ignore
from werkzeug.utils import secure_filename #type: ignore
import os
import uuid
import json
import time
import random
from datetime import datetime
from PyPDF2 import PdfReader #type: ignore
from celery import Celery #type: ignore
import requests #type: ignore
from openai import OpenAI #type: ignore
from dotenv import load_dotenv #type: ignore
import google.generativeai as genai #type: ignore
from collections import defaultdict
import json
request_tracker = defaultdict(list)  # Tracks request_id -> task_ids

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1024*1024*5, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config.update(
    UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
    MAX_CONTENT_LENGTH=50*1024*1024,
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    CELERY_RESULT_EXPIRES=300,
    CELERY_TASK_IGNORE_RESULT=False
)

# Initialize Celery
celery = Celery(
    app.name,
    broker=app.config['CELERY_BROKER_URL'],
    backend=app.config['CELERY_RESULT_BACKEND']
)
celery.conf.update(
    result_extended=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    broker_connection_retry_on_startup=True,
    worker_concurrency=2,
    task_acks_late=True
)

# Load environment variables
load_dotenv()
oaiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
import typing_extensions as typing #type: ignore

class Recipe(typing.TypedDict):
    recipe_name: str
    ingredients: list[str]
geminimodel = genai.GenerativeModel("gemini-exp-1206")
# Constants
#debug:
MAX_API_TIMEOUT = 45
MAX_TEXT_LENGTH = 1200000
API_RETRY_DELAYS = [5, 15, 45]
RPM_LIMIT = 3500

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def log_request(request_id, text, prompt_prefix, summaries):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'prompt_prefix': prompt_prefix,
        'text_preview': text[:200] + '...' if len(text) > 200 else text,
        'summaries': summaries,
        'rankings': {},
        'quality_scores': {}
    }
    
    try:
        # Append new entry instead of searching
        with open('requests.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Initial log failed: {str(e)}")



def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''.join(page.extract_text() for page in reader.pages)
        logger.info(f"Extracted {len(text)} characters from PDF")
        wordcount = len(text.split(" "))
        logger.info(f"Extracted {wordcount} words from PDF")
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise

@celery.task(
    bind=True,
    max_retries=3,
    time_limit=MAX_API_TIMEOUT*2,
    soft_time_limit=MAX_API_TIMEOUT+5,
    rate_limit=f"{RPM_LIMIT//60}/s"
)
def process_summary(self, text, prompt_prefix, model, request_id=None):
    try:
        logger.info(f"Starting {model} processing (attempt {self.request.retries + 1})")
        session = requests.Session()
        endpoint = None
        headers = {}
        payload = {}
        logger.info("Text Used: " + text[min(len(text), MAX_TEXT_LENGTH)-100:min(len(text), MAX_TEXT_LENGTH)])
        if model == "openai":
            #Nikolas note: OpenAI needs their client package, let's use that
            #Nikolas note: [2025-01-23 14:20:02,258: ERROR/ForkPoolWorker-1] openai error: Error code: 429 - {'error': {'message': 'Request too large for gpt-4o in organization org-FPY5TvqSJGKGWU6QsfNtWZB7 on tokens per min (TPM): Limit 30000, Requested 34102. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}. Retrying in 16.37188766227481s - Jesus christ what an annoyance.
            completion = oaiclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt_prefix},
                    {
                        "role": "user",
                        "content": text[:min(len(text), MAX_TEXT_LENGTH)]
                    }
                ]
            )
            logger.info("OpenAI completion: " + str(completion))

        elif model == "deepseek":
            endpoint = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            payload = {
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user", 
                    "content": f"{prompt_prefix}\n\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}"
                }]
            }

        elif model == "claude":
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": f"{prompt_prefix}\n\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}"
                }]
            }
        elif model == "perplexity":  # New section
            endpoint = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "sonar-pro",
                "messages": [{
                    "role": "user",
                    "content": f"{prompt_prefix}\n\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}"
                }]
            }
        elif model == "llama3":
            endpoint = 'https://api.llama-api.com/chat/completions'
            headers = {
                "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3.3-70b",
                "messages": [{
                    "role": "user",
                    "content": f"{prompt_prefix}\n\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}"
                }],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False,
            }
        elif model == "grok2":
            endpoint = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {
                    "role": "user",
                    "content": f"{prompt_prefix}\n\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}"
                }],
                "model": "grok-2-latest",
                "temperature": 0,
                "stream": False,
            }
        elif model == "gemini":
            geminiresponse = geminimodel.generate_content(f"{prompt_prefix}\n\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}")

        #The weirdos
        if model == "openai":
            response_json = json.loads(completion.json())
        elif model == "gemini":
            response_json = geminiresponse
        else:
            response = session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=(3.05, MAX_API_TIMEOUT)
            )
            response.raise_for_status()
            response_json = response.json()
        if model == "openai":
            if not response_json.get('choices'):
                raise ValueError("Empty OpenAI response")
            content = response_json['choices'][0]['message']['content']
        elif model == "deepseek":
            content = response_json['choices'][0]['message']['content']
        elif model == "claude":
            content = response_json['content'][0]['text']
        elif model == "gemini":
            #This is also a json
            content = response_json.text
            logger.info("Gemini response: " + content)
            if isinstance(content, list):
                content = "\n".join(content)
        elif model == "perplexity":  # New response handling
            content = response_json['choices'][0]['message']['content']
        elif model == "llama3":
            content = response_json['choices'][0]['message']['content']
        elif model == "grok2":
            content = response_json['choices'][0]['message']['content']
        return {
            'model': model,
            'summary': content,
            'status': 'success',
            'model_id': f"{model}_{uuid.uuid4().hex[:4]}",
            'request_id': request_id  # Include request_id in result
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get('Retry-After', random.choice(API_RETRY_DELAYS)))
            logger.warning(f"Rate limited. Retrying in {retry_after}s")
            raise self.retry(exc=e, countdown=retry_after)
        raise

    except Exception as e:
        retry_index = min(self.request.retries, len(API_RETRY_DELAYS)-1)
        delay = API_RETRY_DELAYS[retry_index] + random.uniform(0, 2)
        logger.error(f"{model} error: {str(e)}. Retrying in {delay}s")
        raise self.retry(exc=e, countdown=delay)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    request_id = str(uuid.uuid4())
    try:
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        prompt_prefix = request.form.get('prompt_prefix', 'Summarize this academic paper:')

        models = ['deepseek','gemini','claude','perplexity','llama3','grok2']
        tasks = [process_summary.apply_async(
            args=(text, prompt_prefix, model,request_id),
        ) for model in models]

        # Store task IDs with request ID
        request_tracker[request_id] = [task.id for task in tasks]

        # Initial log with empty summaries
        log_request(request_id, text, prompt_prefix, [])
        
        return jsonify({
            "request_id": request_id,
            "status_urls": [task.id for task in tasks],
            "message": "Processing started. Check individual model statuses."
        }), 202

    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route('/status/<task_id>')
def task_status(task_id):
    task = process_summary.AsyncResult(task_id)
    
    # Force refresh for completed tasks
    if task.ready():
        task.backend.get_task_meta(task.id)
    
    response = {
        'task_id': task_id,
        'status': task.state.lower(),
        'model': None,
        'summary': None,
        'error': None
    }

    if task.successful():
        result = task.result
        response.update({
            'model': result.get('model'),
            'summary': result.get('summary'),
            'status': 'completed'
        })
        
        # Update log with summary using request_id from task result
        request_id = result.get('request_id')
        if request_id:
            updated = False
            with open('requests.log', 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                f.truncate()
                
                for line in lines:
                    try:
                        log_entry = json.loads(line)
                        if log_entry['request_id'] == request_id:
                            # Initialize summaries list if empty
                            if 'summaries' not in log_entry:
                                log_entry['summaries'] = []
                            # Add new summary
                            log_entry['summaries'].append(result['summary'])
                            updated = True
                        f.write(json.dumps(log_entry) + '\n')
                    except json.JSONDecodeError:
                        continue

    elif task.failed():
        response.update({
            'status': 'failed',
            'error': str(task.result)
        })
    
    return jsonify(response)


@app.route('/rankings', methods=['POST'])
def save_rankings():
    try:
        data = request.json
        request_id = data.get('request_id')
        rankings = data.get('rankings', {})
        quality_scores = data.get('quality_scores', {})

        updated = False
        with open('requests.log', 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()
            
            for line in lines:
                try:
                    log_entry = json.loads(line)
                    if log_entry['request_id'] == request_id:
                        # Check if summaries exist
                        model_count = len(log_entry.get('summaries', []))
                        if model_count == 0:
                            raise ValueError("Summaries not yet generated")
                            
                        # Validate counts match
                        if len(rankings) != model_count:
                            raise ValueError(
                                f"Expected {model_count} rankings, got {len(rankings)}"
                            )
                            
                        if len(quality_scores) != model_count:
                            raise ValueError(
                                f"Expected {model_count} scores, got {len(quality_scores)}"
                            )

                        # Validate ranks
                        unique_ranks = set(rankings.values())
                        if len(unique_ranks) != model_count:
                            raise ValueError("All ranks must be unique")
                            
                        if any(rank < 1 or rank > model_count for rank in unique_ranks):
                            raise ValueError(f"Ranks must be between 1 and {model_count}")

                        log_entry['rankings'] = rankings
                        log_entry['quality_scores'] = quality_scores
                        updated = True
                    
                    f.write(json.dumps(log_entry) + '\n')
                except json.JSONDecodeError:
                    continue

        return jsonify({'status': 'success' if updated else 'not found'})

    except Exception as e:
        logger.error(f"Ranking update failed: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
