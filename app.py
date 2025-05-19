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
import hashlib
import shutil
import redis #type: ignore
from flask_login import ( #type: ignore
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from auth import auth_bp, User, init_login_manager
from werkzeug.security import generate_password_hash, check_password_hash #type: ignore
import typing_extensions as typing #type: ignore
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
    PDF_STORAGE_FOLDER=os.path.join(os.getcwd(), 'pdf_storage'),
    MAX_CONTENT_LENGTH=50*1024*1024,
    CELERY_BROKER_URL=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_EXPIRES=300,
    CELERY_TASK_IGNORE_RESULT=False
)

# Initialize Redis client
redis_client = redis.from_url(app.config['CELERY_BROKER_URL'])

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
    broker_pool_limit=None
)

# Load environment variables
load_dotenv()
oaiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
oaiclient2 = OpenAI(api_key=os.getenv('OPENAI_API_KEY2'))
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PERPLEXITY_API_KEY2 = os.getenv('PERPLEXITY_API_KEY2')


# Initialize gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class Recipe(typing.TypedDict):
    recipe_name: str
    ingredients: list[str]

geminimodel = genai.GenerativeModel("gemini-exp-1206")

# Constants
prompt_suffix = 'Please keep your answer to less than 200 words.'
MAX_API_TIMEOUT = 45
MAX_TEXT_LENGTH = 1200000
API_RETRY_DELAYS = [5, 15, 45]
RPM_LIMIT = 3500

# Create upload directory and PDF storage directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_STORAGE_FOLDER'], exist_ok=True)

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
        'username': username,
        'nickname': nickname,
        'prompt_prefix': prompt_prefix,
        'question_difficulty': question_difficulty,
        'text_preview': text[:200] + '...' if len(text) > 200 else text,
        'summaries': {},
        'rankings': {},
        'quality_scores': {}
    }
    
    try:
        with open('requests.log', 'a') as f:
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
    max_retries=2,
    time_limit=MAX_API_TIMEOUT*2,
    soft_time_limit=MAX_API_TIMEOUT+5,
    rate_limit=f"{RPM_LIMIT//60}/s",
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_jitter=True
)
def process_summary(self, text, prompt_prefix, model, request_id=None, display_name=None, nickname=None):
    try:
        logger.info(f"Starting {model} processing (attempt {self.request.retries + 1})")
        session = requests.Session()
        endpoint = None
        headers = {}
        payload = {}
        logger.info("Text Used: " + text[min(len(text), MAX_TEXT_LENGTH)-100:min(len(text), MAX_TEXT_LENGTH)])
        
        if model == "openai":
            # try primary client
            try:
                completion = oaiclient.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system","content": prompt_prefix},
                              {"role": "user","content": text[:MAX_TEXT_LENGTH]}]
                )
                resp = json.loads(completion.json())
                if not resp.get('choices'):
                    raise ValueError("Empty primary OpenAI response")
            except Exception as e:
                logger.warning(f"Primary OpenAI failed: {e}, retrying with fallback client")
                completion = oaiclient2.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role":"system","content":prompt_prefix},
                              {"role":"user","content": text[:MAX_TEXT_LENGTH]}]
                )
                resp = json.loads(completion.json())
            response_json = resp

        elif model == "deepseek":
            endpoint = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            payload = {
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user", 
                    "content": f"{prompt_prefix}\n{prompt_suffix}\n{ text[:min(len(text),MAX_TEXT_LENGTH)]}"
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
        elif model == "perplexity":
            # try primary key
            headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY2}"}
            for key in (PERPLEXITY_API_KEY, PERPLEXITY_API_KEY2):
                headers["Authorization"] = f"Bearer {key}"
                try:
                    response = session.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=headers,
                        json={
                            "model":"sonar-pro",
                            "messages":[{"role":"user",
                                         "content":f"{prompt_prefix}\n{text[:MAX_TEXT_LENGTH]}"}]
                        },
                        timeout=(3.05, MAX_API_TIMEOUT)
                    )
                    response.raise_for_status()
                    resp = response.json()
                    if resp.get('choices'):
                        response_json = resp
                        break
                except Exception as e:
                    logger.warning(f"Perplexity with key {key} failed: {e}")
            else:
                raise ValueError("Both Perplexity keys failed")
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

        # Handle model responses
        if model == "openai":
            response_json = json.loads(completion.json())
        elif model == "gemini":
            response_json = geminiresponse
        elif model == "perplexity":
            # already set response_json in the Perplexity branch above
            pass
        else:
            # all other models use endpoint/payload
            response = session.post(
                endpoint,  # type: ignore
                headers=headers,
                json=payload,
                timeout=(3.05, MAX_API_TIMEOUT)
            )
            response.raise_for_status()
            response_json = response.json()
            
        # Extract content based on model
        if model == "openai":
            if not response_json.get('choices'):
                raise ValueError("Empty OpenAI response")
            content = response_json['choices'][0]['message']['content']
        elif model == "deepseek":
            content = response_json['choices'][0]['message']['content']
        elif model == "claude":
            content = response_json['content'][0]['text']
        elif model == "gemini":
            content = response_json.text
            logger.info("Gemini response: " + content)
            if isinstance(content, list):
                content = "\n".join(content)
        elif model == "perplexity":
            content = response_json['choices'][0]['message']['content']
        elif model == "llama3":
            content = response_json['choices'][0]['message']['content']
        elif model == "grok2":
            content = response_json['choices'][0]['message']['content']
        
        # Store result in Redis using the REAL model name
        result_data = {
            'model': model,                     # was display_name
            'summary': content,
            'status': 'success',
            'request_id': request_id,
            'nickname': nickname,
            'timestamp': datetime.now().isoformat()
        }
        result_key = f"result:{request_id}:{model}"
        redis_client.setex(result_key, 3600, json.dumps(result_data))

        # Log to file for compatibility, also tag by real model
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'nickname': nickname,
            'model': model,                     # was display_name
            'summary': content
        }
        try:
            with open('requests_questions.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            logger.error("Logging model response failed")
        return result_data

    except Exception as e:
        retry_index = min(self.request.retries, len(API_RETRY_DELAYS)-1)
        delay = API_RETRY_DELAYS[retry_index] + random.uniform(0, 2)
        logger.error(f"{model} error: {str(e)}. Retrying in {delay}s")
        raise self.retry(exc=e, countdown=delay)

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

        # choose real model names only
        all_models = ['perplexity', 'openai']
        selected_models = random.sample(all_models, 2)
        tasks = []
        for model in selected_models:
            task = process_summary.apply_async(
                args=(text, prompt_prefix, model, request_id, None, nickname)
            )
            tasks.append(task)
            request_tracker[request_id].append({
                'task_id': task.id,
                'real_model': model
            })

        log_request(
            request_id=request_id,
            text=text,
            prompt_prefix=prompt_prefix,
            summaries=[],
            question_difficulty=question_difficulty,
            username=current_user.id,
            nickname=nickname
        )
        
        # Log the question details
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
    elif task.failed():
        response.update({
            'status': 'failed',
            'error': str(task.result)
        })
    
    return jsonify(response)

@app.route('/get_answers', methods=['GET'])
@login_required
def get_answers():
    nickname = request.args.get('nickname', '')
    extra = request.args.get('extra', '0')
    try:
        extra = int(extra)
    except ValueError:
        extra = 0
    
    logger.info(f"get_answers called with nickname={nickname}, extra={extra}")

    # First, get questions from file
    questions = []
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'prompt' in entry:  # This is a question entry
                        questions.append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.warning("requests_questions.log file not found")
        return jsonify([])

    # Now get answers from Redis
    user_results = []
    extra_results = []
    
    for question in questions:
        request_id = question['request_id']
        
        # Get all answers for this request from Redis
        pattern = f"result:{request_id}:*"
        model_answers = {}
        
        for key in redis_client.scan_iter(match=pattern):
            result_data = redis_client.get(key)
            if result_data:
                result = json.loads(result_data)
                model_answers[result['model']] = result['summary']
        
        # Only include questions that have at least 2 answers
        if len(model_answers) >= 2:
            item = {
                'request_id': request_id,
                'prompt': question.get('prompt', 'No prompt available'),
                'file': question.get('file', 'File: Unavailable'),
                'model_answers': model_answers
            }
            
            # Separate user questions from potential extra questions
            question_nickname = question.get('nickname')
            if question_nickname == nickname:
                user_results.append(item)
                logger.debug(f"Added user question: {request_id}")
            else:
                extra_results.append(item)
                logger.debug(f"Added to extra questions pool: {request_id} from {question_nickname}")
    
    # If extra > 0, randomly select that many extra questions
    selected_extras = []
    if extra > 0 and extra_results:
        # Ensure we're selecting random items without repeating
        num_to_select = min(extra, len(extra_results))
        selected_extras = random.sample(extra_results, num_to_select)
        logger.info(f"Selected {len(selected_extras)} extra questions out of {len(extra_results)} available")
    
    # Combine into two lists
    return jsonify({
        'user_questions': user_results,
        'extra_questions': selected_extras
    })

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
                    if 'prompt' in entry:  # Only get question entries, not answer entries
                        questions.append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        questions = []
        
    user_questions = [q for q in questions if q.get('nickname') == nickname]
    extra_questions = []
    
    # Only select extra questions if explicitly requested
    if extra > 0:
        # Filter out questions that belong to the user
        other_questions = [q for q in questions if q.get('nickname') != nickname]
        if other_questions:
            # Randomly select up to 'extra' number of questions
            extra_questions = random.sample(other_questions, min(extra, len(other_questions)))
            
    return jsonify({
        'user_questions': user_questions,
        'extra_questions': extra_questions
    })

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
            with open('requests_ratings.log', 'a') as f:
                f.write(json.dumps(entry) + '\n')
            saved.append(request_id)
        except Exception as e:
            logger.error(f"Saving rating failed for {request_id}: {str(e)}")
            
    return jsonify({'status': 'success', 'saved': saved})

@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return render_template('index.html')

@app.route('/rankings', methods=['POST'])
@app.route('/rankings/', methods=['POST'])
@login_required
def save_rankings():
    data = request.json
    request_id = data.get('request_id')
    rankings = data.get('rankings', {})
    quality_scores = data.get('quality_scores', {})
    model_answers = data.get('model_answers', {})

    answer_entry = {
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'nickname': current_user.id,
        'rankings': rankings,
        'quality_scores': quality_scores,
        'model_answers': model_answers
    }
    try:
        with open('requests_answers.log', 'a') as f:
            f.write(json.dumps(answer_entry) + '\n')
    except Exception as e:
        logger.error(f"Ranking update failed: {str(e)}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500

    logger.info(f"Successfully saved rankings for request {request_id}")
    return jsonify({'status': 'success'})

@app.route('/leaderboard', methods=['GET'])
@login_required
def leaderboard():
    # map request_id → difficulty
    diff_map = {}
    try:
        with open('requests_questions.log','r') as fq:
            for line in fq:
                e = json.loads(line)
                if 'request_id' in e and 'question_difficulty' in e:
                    diff_map[e['request_id']] = e['question_difficulty']
    except FileNotFoundError:
        pass

    # aggregate per real_model
    agg = defaultdict(lambda: {'quality_scores': [], 'wins': 0})
    sel_diff = request.args.get('difficulty','').strip()
    try:
        with open('requests_answers.log','r') as fa:
            for line in fa:
                e = json.loads(line)
                rid = e.get('request_id')
                if sel_diff and diff_map.get(rid) != sel_diff:
                    continue

                # Use the real_model field under 'summaries'
                if 'summaries' in e:
                    for disp, info in e['summaries'].items():
                        real = info.get('real_model')
                        # append quality score
                        score = e.get('quality_scores', {}).get(disp)
                        if score is not None:
                            agg[real]['quality_scores'].append(score)
                        # count wins
                        if e.get('rankings', {}).get(disp) == 1:
                            agg[real]['wins'] += 1

    except FileNotFoundError:
        pass

    models = []
    for name, stats in agg.items():
        models.append({
            'name': name,
            'quality_scores': stats['quality_scores'],
            'wins': stats['wins']
        })
    return jsonify({'models': models})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)