import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template, redirect, url_for #type: ignore
from werkzeug.utils import secure_filename #type: ignore
import os
import uuid
import json
import time
import random
from datetime import datetime
import pymupdf4llm #type: ignore
from celery import Celery #type: ignore
import requests #type: ignore
from openai import OpenAI #type: ignore
from dotenv import load_dotenv #type: ignore
import google.generativeai as genai #type: ignore
from collections import defaultdict
import json
import hashlib  # Add for MD5 hashing
import shutil  # Add for file operations
#Adding logins
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from auth import auth_bp, User, init_login_manager  # Remove login_manager from import

from werkzeug.security import generate_password_hash, check_password_hash
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
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Initialize the login manager with user loader
init_login_manager(login_manager)

# Register blueprint
app.register_blueprint(auth_bp)

# Configuration
app.config.update(
    UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
    PDF_STORAGE_FOLDER=os.path.join(os.getcwd(), 'pdf_storage'),  # Add permanent storage folder
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
    task_acks_late=True,
    broker_pool_limit=None  # Add this line
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

# Create upload directory and PDF storage directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_STORAGE_FOLDER'], exist_ok=True)

import threading

log_lock = threading.Lock()

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_duplicate_pdf(file_path):
    """Check if a PDF with the same MD5 hash exists in storage"""
    file_md5 = calculate_md5(file_path)
    
    for filename in os.listdir(app.config['PDF_STORAGE_FOLDER']):
        if filename.endswith('.pdf'):
            stored_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], filename)
            if calculate_md5(stored_path) == file_md5:
                return filename
    return None

def log_request(request_id, text, prompt_prefix, summaries, question_difficulty, username, nickname):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'username': username,  # Add username
        'nickname': nickname,  # Add nickname
        'prompt_prefix': prompt_prefix,
        'question_difficulty': question_difficulty,  # Add question difficulty
        'text_preview': text[:200] + '...' if len(text) > 200 else text,
        'summaries': {},  # Will store {display_name: {real_model: ..., summary: ...}}
        'rankings': {},
        'quality_scores': {}
    }
    
    try:
        log_file_path = os.path.join(app.root_path, 'requests.log')
        with log_lock:
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Initial log failed: {str(e)}")



def extract_text_from_pdf(pdf_path):
    try:
        text = pymupdf4llm.to_markdown(pdf_path)
        logger.info(f"Extracted {len(text)} characters from PDF")
        wordcount = len(text.split(" "))
        logger.info(f"Extracted {wordcount} words from PDF")
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise

@celery.task(
    bind=True,
    max_retries=2,  # Reduced from 3
    time_limit=MAX_API_TIMEOUT*2,
    soft_time_limit=MAX_API_TIMEOUT+5,
    rate_limit=f"{RPM_LIMIT//60}/s",
    autoretry_for=(Exception,),  # Add automatic retry
    retry_backoff=5,  # Add exponential backoff
    retry_jitter=True
)
def process_summary(self, text, prompt_prefix, model, request_id=None, display_name=None, nickname=None):  # renamed parameter
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
            'model': display_name,  # Show display name to users
            'real_model': model,    # Keep actual model for internal tracking
            'summary': content,
            'status': 'success',
            'model_id': f"{model}_{uuid.uuid4().hex[:4]}",
            'request_id': request_id
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

    finally:
        # Log the model's response to requests_questions.log
        if 'content' in locals():
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id,
                'nickname': nickname,  # Use the provided nickname
                'model': display_name,
                'real_model': model,
                'summary': content
            }
            try:
                with open('requests_questions.log', 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                logger.error(f"Logging model response failed: {str(e)}")

# New logging helper for questions
def log_question(request_id, text, prompt_prefix, question_difficulty, nickname, filename):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'nickname': nickname,
        'prompt': prompt_prefix,
        'question_difficulty': question_difficulty,
        'file': filename,
        'text_preview': text[:200] + '...' if len(text) > 200 else text
    }
    try:
        with open('requests_questions.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        logger.error(f"Question log failed: {str(e)}")

@app.route('/available_pdfs', methods=['GET'])
@login_required
def get_available_pdfs():
    """Get list of available PDFs in storage"""
    try:
        pdf_files = []
        for filename in os.listdir(app.config['PDF_STORAGE_FOLDER']):
            if filename.endswith('.pdf'):
                file_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], filename)
                pdf_files.append({
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        
        # Sort by modified date, newest first
        pdf_files.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'pdfs': pdf_files})
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
@login_required
def summarize():
    request_id = str(uuid.uuid4())
    try:
        # Check if user selected an existing PDF
        selected_pdf = request.form.get('selected_pdf')
        
        if selected_pdf and selected_pdf != 'upload':
            # User selected an existing PDF
            pdf_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], selected_pdf)
            if not os.path.exists(pdf_path):
                return jsonify({"error": "Selected PDF not found"}), 404
            filename = selected_pdf
        else:
            # User is uploading a new PDF
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            if not os.path.exists(temp_path):
                return jsonify({"error": "File upload failed"}), 500

            # Check for duplicate
            duplicate_filename = find_duplicate_pdf(temp_path)
            
            if duplicate_filename:
                # Use existing file
                pdf_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], duplicate_filename)
                os.remove(temp_path)  # Remove temporary upload
                filename = duplicate_filename
                logger.info(f"Duplicate PDF detected. Using existing file: {duplicate_filename}")
            else:
                # Move to permanent storage
                pdf_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], filename)
                shutil.move(temp_path, pdf_path)
                logger.info(f"New PDF stored: {filename}")

        text = extract_text_from_pdf(pdf_path)
        prompt_prefix = request.form.get('prompt_prefix', 'Summarize this academic paper:')
        question_difficulty = request.form.get('question_difficulty', 'Medium')
        
        # Get nickname from form and validate
        nickname = request.form.get('nickname', '')
        if not nickname.strip():
            return jsonify({"error": "Nickname cannot be empty"}), 400

        #all_models = ['deepseek','gemini','claude','perplexity','llama3','grok2','openai']
        all_models = ['perplexity','openai']
        selected_models = random.sample(all_models, 2)
        display_names = [f"Model {i+1}" for i in range(2)]  # More descriptive names
        
        tasks = []
        for model, display_name in zip(selected_models, display_names):
            task = process_summary.apply_async(
                args=(text, prompt_prefix, model, request_id, display_name, nickname)  # Pass provided nickname
            )
            tasks.append(task)
            request_tracker[request_id].append({
                'task_id': task.id,
                'real_model': model,
                'display_name': display_name
            })

        log_request(
            request_id=request_id,
            text=text,
            prompt_prefix=prompt_prefix,
            summaries=[],
            question_difficulty=question_difficulty,
            username=current_user.id,  # Add current user's username
            nickname=nickname  # Add nickname to log
        )
        
        # Log the question details into a dedicated log file
        log_question(request_id, text, prompt_prefix, question_difficulty, nickname, filename)
        return jsonify({
            "request_id": request_id,
            "status_urls": [task.id for task in tasks],
            "message": "Processing started. Check individual model statuses."
        }), 202

    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/status/<task_id>')
@login_required
def task_status(task_id):
    task = process_summary.AsyncResult(task_id)
    
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
        
        request_id = result.get('request_id')
        if request_id:
            # Asynchronously update requests_questions.log with the model response
            update_question_log.delay(
                request_id,
                result.get('model'),
                result.get('real_model'),
                result.get('summary'),
                current_user.id
            )
    elif task.failed():
        response.update({
            'status': 'failed',
            'error': str(task.result)
        })
    
    return jsonify(response)

# Remove the first get_answers route definition and keep only this one
@app.route('/get_answers', methods=['GET'])
@login_required
def get_answers():
    nickname = request.args.get('nickname', '')
    extra = request.args.get('extra', '0')
    try:
        extra = int(extra)
    except ValueError:
        extra = 0

    entries = []
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify([])

    # Group entries by request_id; each group should have a question and its answers.
    groups = defaultdict(lambda: {'question': None, 'answers': []})
    for entry in entries:
        req_id = entry.get('request_id')
        if not req_id:
            continue
        if 'prompt' in entry:
            groups[req_id]['question'] = entry
            groups[req_id]['nickname'] = entry.get('nickname')
        else:
            groups[req_id]['answers'].append(entry)

    # Build two lists: one for user questions and one for others (only include complete groups)
    user_questions = []
    other_questions = []
    for req_id, data in groups.items():
        if not data['question'] or len(data['answers']) == 0:
            continue
        item = {
            'request_id': req_id,
            'prompt': data['question'].get('prompt', 'No prompt available'),
            'file': data['question'].get('file', 'File: Unavailable'),
            'model_answers': {
                ans.get('model'): ans.get('summary', 'No answer available')
                for ans in data['answers']
            }
        }
        if data.get('nickname', '') == nickname:
            user_questions.append(item)
        else:
            other_questions.append(item)

    # Randomly sample extra questions from other_users if needed
    extra_questions = []
    if extra > 0 and other_questions:
        extra_questions = random.sample(other_questions, min(extra, len(other_questions)))

    result = user_questions + extra_questions
    return jsonify(result)
    
@app.route('/rankings', methods=['POST'])
@login_required
def save_rankings():
    try:
        data = request.json
        request_id = data.get('request_id')
        rankings = data.get('rankings', {})
        quality_scores = data.get('quality_scores', {})
        model_answers = data.get('model_answers', {})

        # First read the question entry to get all metadata
        question_entry = None
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('request_id') == request_id:
                        question_entry = entry
                        break
                except json.JSONDecodeError:
                    continue

        if not question_entry:
            raise ValueError("Original question not found")

        # Count the number of model answers in requests_questions.log
        model_count = 0
        summaries = {}
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('request_id') == request_id and 'model' in entry:
                        model_count += 1
                        summaries[entry['model']] = {
                            'real_model': entry['real_model'],
                            'summary': entry['summary']
                        }
                except json.JSONDecodeError:
                    continue

        if model_count < 2:
            logger.warning(f"Not enough model answers found for request {request_id}")
            return jsonify({'status': 'failed', 'error': 'Not enough model answers found'}), 400

        # Save to requests_answers.log
        answer_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'nickname': current_user.id,
            'prompt': question_entry.get('prompt'),
            'file': question_entry.get('file'),
            'question_difficulty': question_entry.get('question_difficulty'),  # New: log question difficulty
            'summaries': summaries,
            'rankings': rankings,
            'quality_scores': quality_scores
        }

        with open('requests_answers.log', 'a') as f:
            f.write(json.dumps(answer_entry) + '\n')

        updated = True
        logger.info(f"Successfully saved rankings for request {request_id}")
        return jsonify({'status': 'success'})

        updated = False
        log_file_path = os.path.join(app.root_path, 'requests.log')
        try:
            with open(log_file_path, 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                f.truncate()

                for line in lines:
                    try:
                        log_entry = json.loads(line)
                        if log_entry['request_id'] == request_id:
                            # Check if summaries exist
                            if not log_entry.get('summaries') or len(log_entry['summaries']) < 2:
                                logger.warning(f"Not enough summaries found for request {request_id}")
                                continue

                            model_count = len(log_entry['summaries'])
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
        except Exception as e:
            logger.error(f"Error updating rankings in requests.log: {str(e)}")
            return jsonify({'status': 'failed', 'error': str(e)}), 500

        if updated:
            logger.info(f"Successfully updated rankings for request {request_id}")
            return jsonify({'status': 'success'})
        else:
            logger.warning(f"Request {request_id} not found in requests.log")
            return jsonify({'status': 'not found'}), 404

    except Exception as e:
        logger.error(f"Ranking update failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_questions', methods=['GET'])
@login_required
def get_questions():
    nickname = request.args.get('nickname', '')
    extra = request.args.get('extra', '0')
    try:
        extra = int(extra)
    except ValueError:
        extra = 0
    questions = []
    # Read questions from requests_questions.log
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    questions.append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        questions = []
    user_questions = [q for q in questions if q.get('nickname') == nickname]
    extra_questions = []
    if extra > 0:
        pool = [q for q in questions if q.get('nickname') != nickname]
        if pool:
            extra_questions = random.sample(pool, min(extra, len(pool)))
    return jsonify({
        'user_questions': user_questions,
        'extra_questions': extra_questions
    })

# In save_ratings, save ratings into a new file so that requests_questions.log remains free of user ratings.
@app.route('/save_ratings', methods=['POST'])
@login_required
def save_ratings():
    data = request.json
    ratings = data.get('ratings', {})
    saved = []
    for request_id, score in ratings.items():
        entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'nickname': current_user.id,
            'rating': score
        }
        try:
            with open('requests_ratings.log', 'a') as f:  # <-- using a separate file for ratings
                f.write(json.dumps(entry) + '\n')
            saved.append(request_id)
        except Exception as e:
            logger.error(f"Saving rating failed for {request_id}: {str(e)}")
    return jsonify({'status': 'success', 'saved': saved})

@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))  # Update to use blueprint route
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)