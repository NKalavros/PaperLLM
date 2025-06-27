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
login_manager.login_view = 'auth.login' #type: ignore

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

# --- new: gather multiple API keys/clients ---
OPENAI_API_KEYS = [os.getenv(f'OPENAI_API_KEY{i}') for i in range(1,6)]
OPENAI_API_KEYS = [k for k in OPENAI_API_KEYS if k]
OPENAI_CLIENTS = [OpenAI(api_key=k) for k in OPENAI_API_KEYS]

PERPLEXITY_API_KEYS = [os.getenv(f'PERPLEXITY_API_KEY{i}') for i in range(1,6)]
PERPLEXITY_API_KEYS = [k for k in PERPLEXITY_API_KEYS if k]
# --- end new ---

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
prompt_suffix = 'Make sure your answers are 5 sentences or less. Ensure that your answer contains information from the above provided text.'
MAX_API_TIMEOUT = 45
MAX_TEXT_LENGTH = 1200000
API_RETRY_DELAYS = [5, 15, 45]
RPM_LIMIT = 3500

# Create upload directory and PDF storage directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_STORAGE_FOLDER'], exist_ok=True)

# Add predefined talks mapping after the constants section
PREDEFINED_TALKS = {
    'Dr. Julio Saez Rodriguez - "Benchmarking foundation models in biology: where we are, and where we want to go with the community"': 'talk1.pdf',
    'Keynote Presentation: Dr. Bo Wang (Univ of Toronto) – "Building Foundation Models for Single-cell Omics and Imaging"': 'talk2.pdf', 
    'Dr. Maria Brbic – "Predicting Perturbation Effects: Are We Really There?"': 'talk3.pdf',
    'Dr. Pablo Meyer Rojas – "The AI Alliance and the benchmarking of foundation models for drug discovery"': 'talk4.pdf',
    'Dr. Katrina Kalantar – "Benchmarking in Service of Virtual Cell Models: Challenges, Opportunities, and a Path Forward"': 'talk5.pdf',
    'Dr. Anshul Kundaje - "Deep learning models of regulatory DNA: A critical analysis of model design choices"': 'talk6.pdf',
    'Dr. Justin Guinney (Tempus AI) – "Benchmarking Multi-Modal Large Language Models for Metastatic Breast Cancer Prognosis"': 'talk7.pdf'
}

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

@app.route('/available_pdfs', methods=['GET'])
@login_required
def get_available_pdfs():
    """Get list of available PDFs in storage, including predefined talks"""
    try:
        pdf_files = []
        
        # Add predefined talks (always show them, mark if PDF exists)
        for talk_title, filename in PREDEFINED_TALKS.items():
            file_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], filename)
            pdf_exists = os.path.exists(file_path)
            
            pdf_files.append({
                'filename': filename,
                'display_name': talk_title,
                'size': os.path.getsize(file_path) if pdf_exists else 0,
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat() if pdf_exists else None,
                'is_predefined': True,
                'pdf_exists': pdf_exists
            })
        
        # Add other uploaded PDFs
        for filename in os.listdir(app.config['PDF_STORAGE_FOLDER']):
            if filename.endswith('.pdf') and filename not in PREDEFINED_TALKS.values():
                file_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], filename)
                pdf_files.append({
                    'filename': filename,
                    'display_name': filename,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    'is_predefined': False,
                    'pdf_exists': True
                })
        
        # Sort predefined talks first, then by modified date
        pdf_files.sort(key=lambda x: (not x['is_predefined'], x['modified'] or ''), reverse=True)
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
        # only Gustavo may upload new PDFs
        if not selected_pdf and current_user.id != 'gustavo':
            return jsonify({'error': 'Only Gustavo can upload new PDFs'}), 403

        if selected_pdf and selected_pdf != 'upload':
             # User selected an existing PDF
            pdf_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], selected_pdf)
            
            # Check if it's a predefined talk
            is_predefined = selected_pdf in PREDEFINED_TALKS.values()
            
            if not os.path.exists(pdf_path):
                if is_predefined:
                    # For predefined talks, queue the task anyway - it will wait for PDF
                    filename = selected_pdf
                    pdf_path = None  # Signal that PDF doesn't exist yet
                else:
                    return jsonify({"error": "Selected PDF not found"}), 404
            filename = selected_pdf
        else:
            # User is uploading a new PDF
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            filename = secure_filename(file.filename) #type: ignore
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

        # Extract text only if PDF exists
        if pdf_path and os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path)
        else:
            text = None  # Will be handled by the Celery task

        prompt_prefix = request.form.get('prompt_prefix', 'Summarize this academic paper:')
        question_difficulty = request.form.get('question_difficulty', 'Easy')
        
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
                args=(text, prompt_prefix, model, request_id, None, nickname, filename)
            )
            tasks.append(task)
            request_tracker[request_id].append({
                'task_id': task.id,
                'real_model': model
            })

        log_request(
            request_id=request_id,
            text=text or f"PDF pending for {filename}",
            prompt_prefix=prompt_prefix,
            summaries=[],
            question_difficulty=question_difficulty,
            username=current_user.id,
            nickname=nickname
        )
        
        # Log the question details
        log_question(request_id, text or f"PDF pending for {filename}", prompt_prefix, question_difficulty, nickname, filename)
        
        status_message = "Processing started. Check individual model statuses."
        if not text:
            status_message += f" Note: PDF {filename} not yet available - task will wait for upload."
            
        return jsonify({
            "request_id": request_id,
            "status_urls": [task.id for task in tasks],
            "message": status_message
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
    raw_nick = request.args.get('nickname', '').strip()
    # strip author_ prefix if present
    if raw_nick.startswith('author_'):
        nickname = raw_nick[len('author_'):]
    else:
        nickname = raw_nick
    nickname = nickname.lower()
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
        qn = question.get('nickname','').strip().lower()
        # match against stripped nickname
        if qn == nickname:
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
                user_results.append(item)
                logger.debug(f"Added user question: {request_id}")
        else:
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
                extra_results.append(item)
                logger.debug(f"Added to extra questions pool: {request_id} from {question.get('nickname')}")
    
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
        
    user_questions = [
        q for q in questions
        if q.get('nickname','').strip().lower() == nickname.strip().lower()
    ]
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
    ratings = data.get('ratings', {}) # type: ignore
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
    return render_template('index.html', username=current_user.id)

@app.route('/rankings', methods=['POST'])
@app.route('/rankings/', methods=['POST'])
@login_required
def save_rankings():
    data = request.json
    request_id = data.get('request_id') # type: ignore
    rankings = data.get('rankings', {}) # type: ignore
    quality_scores = data.get('quality_scores', {}) # type: ignore
    model_answers = data.get('model_answers', {}) # type: ignore
    real_model_mapping = data.get('real_model_mapping', {}) # type: ignore
    is_speaker = data.get('is_speaker', False)    # type: ignore
    
    # Calculate real_rankings and real_quality_scores using real_model_mapping
    real_rankings = {}
    real_quality_scores = {}
    
    # Transform rankings to use real model names (only if rankings were provided)
    if rankings:
        for display_name, rank in rankings.items():
            real_model = real_model_mapping.get(display_name)
            if real_model:
                real_rankings[real_model] = rank
    
    # Transform quality scores to use real model names (skip null/empty values)
    for display_name, score in quality_scores.items():
        real_model = real_model_mapping.get(display_name)
        if real_model and score is not None:
            real_quality_scores[real_model] = score
    
    # Retrieve question difficulty from requests_questions.log
    question_difficulty = "Easy"  # Default if not found
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('request_id') == request_id and 'question_difficulty' in entry:
                        question_difficulty = entry['question_difficulty']
                        break
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.warning("requests_questions.log not found when retrieving question difficulty")

    # Retrieve asker's nickname from requests_questions.log
    asker_nickname = None
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # look only at question entries
                    if entry.get('request_id') == request_id and 'prompt' in entry:
                        asker_nickname = entry.get('nickname')
                        break
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.warning("requests_questions.log not found when retrieving asker nickname")
    
    # Prepend author_ to the asker’s name when logging if is_speaker flag is set
    if asker_nickname and is_speaker:
        asker_nickname = f"author_{asker_nickname}"
    
    # Build and log the answer entry, now including asker_nickname
    answer_entry = {
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'ranker_nickname': current_user.id,
        'asker_nickname': asker_nickname,
        'rankings': rankings,
        'quality_scores': quality_scores,
        'model_answers': {  # truncated versions
            model: ans[:100] + '...' if len(ans) > 100 else ans
            for model, ans in model_answers.items()
        },
        'real_rankings': real_rankings,
        'real_quality_scores': real_quality_scores,
        'question_difficulty': question_difficulty
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
    # map request_id → question_difficulty
    diff_map = {}
    try:
        with open('requests_questions.log','r') as fq:
            for line in fq:
                e = json.loads(line)
                if 'request_id' in e and 'question_difficulty' in e:
                    diff_map[e['request_id']] = e['question_difficulty']
    except FileNotFoundError:
        pass

    # collect scores per model and per difficulty and overall
    agg = defaultdict(lambda: defaultdict(list))  # agg[model][difficulty]
    try:
        with open('requests_answers.log','r') as fa:
            for line in fa:
                e = json.loads(line)
                rid = e.get('request_id')
                
                # First try to get difficulty directly from the answer entry (new format)
                # If not available, fall back to the difficulty map (old format)
                diff = e.get('question_difficulty')
                if diff is None:
                    diff = diff_map.get(rid, 'All')
                
                # Handle all formats:
                # 1. Old format with 'summaries' containing 'real_model'
                # 2. New format with real_quality_scores
                # 3. Transitional format with direct model names in quality_scores
                
                # First try the newest format with real_quality_scores
                if 'real_quality_scores' in e:
                    for model, score in e.get('real_quality_scores', {}).items():
                        # Only include valid numeric scores
                        if score is not None and isinstance(score, (int, float)):
                            agg[model][diff].append(score)
                            agg[model]['All'].append(score)
                
                # Then try the old format with summaries
                elif 'summaries' in e:
                    for disp, info in e.get('summaries', {}).items():
                        real = info.get('real_model')
                        score = e.get('quality_scores', {}).get(disp)
                        # Only include valid numeric scores
                        if score is not None and isinstance(score, (int, float)) and real is not None:
                            agg[real][diff].append(score)
                            agg[real]['All'].append(score)
                
                # Finally try the transitional format with direct model names in quality_scores
                else:
                    for model, score in e.get('quality_scores', {}).items():
                        # Only include valid numeric scores and real model names
                        if score is not None and isinstance(score, (int, float)) and not model.startswith('Model '):
                            agg[model][diff].append(score)
                            agg[model]['All'].append(score)
    except FileNotFoundError:
        pass

    # compute mean and sem
    import math
    def stats_list(lst):
        n = len(lst)
        if n == 0:
            return {'mean': None, 'sem': None}
        m = sum(lst) / n
        var = sum((x - m) ** 2 for x in lst) / n
        sem = math.sqrt(var) / math.sqrt(n)
        return {'mean': round(m, 2), 'sem': round(sem, 2)}

    results = []
    for model, diffs in agg.items():
        results.append({
            'name': model.replace('openai','OpenAI').replace('perplexity','Perplexity'),
            'stats': {
                'Easy': stats_list(diffs.get('Easy', [])),
                'Hard': stats_list(diffs.get('Hard', [])),
                'All': stats_list(diffs.get('All', []))
            }
        })
    
    # compute t-tests per difficulty
    from scipy.stats import ttest_ind #type: ignore
    
    def format_p_value(p):
        """Format p-value with scientific notation"""
        p_rounded = round(p, 2)
        if p > 0.05:
            return {'p_value': p_rounded, 'notation': 'N.S.'}
        elif p > 0.01:
            return {'p_value': p_rounded, 'notation': '*'}
        elif p > 0.001:
            return {'p_value': p_rounded, 'notation': '**'}
        else:
            return {'p_value': p_rounded, 'notation': '***'}
    
    ttest = {}
    model_keys = list(agg.keys())[:2]
    for diff in ('Easy','Hard','All'):
        if len(model_keys)==2:
            x = agg[model_keys[0]][diff]
            y = agg[model_keys[1]][diff]
            if x and y:
                _, p = ttest_ind(x, y, equal_var=False)
                p_formatted = format_p_value(p)
                ttest[diff] = {'N': min(len(x), len(y)), **p_formatted}
    return jsonify({'models': results, 'ttest': ttest})

@app.route('/leaderboard/speaker', methods=['GET'])
@login_required
def speaker_leaderboard():
    # read only answers with asker_nickname starting "author_"
    agg = defaultdict(lambda: defaultdict(list))
    try:
        with open('requests_answers.log','r') as fa:
            for line in fa:
                e = json.loads(line)
                asker = e.get('asker_nickname','')
                if not asker.startswith('author_'):
                    continue
                # same aggregation logic as /leaderboard
                diff = e.get('question_difficulty') or 'All'
                for model, score in e.get('real_quality_scores', {}).items():
                    if isinstance(score,(int,float)):
                        agg[model][diff].append(score)
                        agg[model]['All'].append(score)
    except FileNotFoundError:
        pass

    import math
    def stats_list(lst):
        n=len(lst)
        if n==0: return {'mean':None,'sem':None}
        m=sum(lst)/n
        sem=math.sqrt(sum((x-m)**2 for x in lst)/n)/math.sqrt(n)
        return {'mean':round(m,2),'sem':round(sem,2)}

    results=[]
    for model,diffs in agg.items():
        results.append({
            'name': model.replace('openai','OpenAI').replace('perplexity','Perplexity'),
            'stats': {
                'Easy': stats_list(diffs.get('Easy',[])),
                'Hard': stats_list(diffs.get('Hard',[])),
                'All': stats_list(diffs.get('All',[]))
            }
        })
    
    # compute t-tests per difficulty
    from scipy.stats import ttest_ind # type: ignore
    
    def format_p_value(p):
        """Format p-value with scientific notation"""
        p_rounded = round(p, 2)
        if p > 0.05:
            return {'p_value': p_rounded, 'notation': 'N.S.'}
        elif p > 0.01:
            return {'p_value': p_rounded, 'notation': '*'}
        elif p > 0.001:
            return {'p_value': p_rounded, 'notation': '**'}
        else:
            return {'p_value': p_rounded, 'notation': '***'}
    
    ttest = {}
    model_keys = list(agg.keys())[:2]
    for diff in ('Easy','Hard','All'):
        if len(model_keys)==2:
            x = agg[model_keys[0]][diff]
            y = agg[model_keys[1]][diff]
            if x and y:
                _, p = ttest_ind(x, y, equal_var=False)
                p_formatted = format_p_value(p)
                ttest[diff] = {'N': min(len(x), len(y)), **p_formatted}
    return jsonify({'models': results, 'ttest': ttest})

@app.route('/speaker_talks', methods=['GET'])
@login_required
def speaker_talks():
    """Return list of all talk filenames seen in requests_questions.log"""
    talks = set()
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if 'file' in e:
                        talks.add(e['file'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify({'talks': []})
    return jsonify({'talks': sorted(talks)})

@app.route('/speaker_questions', methods=['GET'])
@login_required
def speaker_questions():
    """
    Return up to `num` random questions (with >=2 answers) for the given talk.
    Query params: talk=<filename>, num=<int>
    """
    talk = request.args.get('talk', '')
    try:
        num = max(0, int(request.args.get('num', '0')))
    except ValueError:
        num = 0

    # collect question entries for this talk
    qs = []
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get('file') == talk and 'prompt' in e:
                        qs.append(e)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify({'questions': []})

    # attach answers from Redis, only keep those with >=2
    results = []
    for q in qs:
        rid = q['request_id']
        pattern = f"result:{rid}:*"
        m_ans = {}
        for key in redis_client.scan_iter(match=pattern):
            data = redis_client.get(key)
            if not data: continue
            r = json.loads(data)
            m_ans[r['model']] = r['summary']
        if len(m_ans) >= 2:
            results.append({
                'request_id': rid,
                'prompt': q.get('prompt'),
                'file': q.get('file'),
                'model_answers': m_ans
            })

    # random subset
    if num > 0 and results:
        results = random.sample(results, min(num, len(results)))
    return jsonify({'questions': results})

# load GitHub repo URL
GITHUB_REPO_URL = os.getenv('GITHUB_REPO_URL', 'https://github.com/your-org/your-repo.git')

# configure periodic clone every 60s
celery.conf.beat_schedule = {
    'periodic-repo-sync': {
        'task': 'app.update_repo',
        'schedule': 60.0
    }
}

@celery.task(name='app.update_repo')
def update_repo():
    tmpdir = tempfile.mkdtemp()
    # clone or pull latest
    subprocess.run(['git', 'clone', GITHUB_REPO_URL, tmpdir], check=True)
    dest = app.config['UPLOAD_FOLDER']
    # clear old uploads
    for name in os.listdir(dest):
        path = os.path.join(dest, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    # copy fresh contents
    for name in os.listdir(tmpdir):
        src = os.path.join(tmpdir, name)
        dst = os.path.join(dest, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    # record sync time
    redis_client.set('repo_last_update', datetime.now().isoformat())

@app.route('/repo_status', methods=['GET'])
@login_required
def repo_status():
    last = redis_client.get('repo_last_update')
    return jsonify({'last_update': last.decode() if last else None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)
        with open('requests_answers.log','r') as fa:
            for line in fa:
                e = json.loads(line)
                asker = e.get('asker_nickname','')
                if not asker.startswith('author_'):
                    continue
                # same aggregation logic as /leaderboard
                diff = e.get('question_difficulty') or 'All'
                for model, score in e.get('real_quality_scores', {}).items():
                    if isinstance(score,(int,float)):
                        agg[model][diff].append(score)
                        agg[model]['All'].append(score)
    except FileNotFoundError:
        pass

    import math
    def stats_list(lst):
        n=len(lst)
        if n==0: return {'mean':None,'sem':None}
        m=sum(lst)/n
        sem=math.sqrt(sum((x-m)**2 for x in lst)/n)/math.sqrt(n)
        return {'mean':round(m,2),'sem':round(sem,2)}

    results=[]
    for model,diffs in agg.items():
        results.append({
            'name': model.replace('openai','OpenAI').replace('perplexity','Perplexity'),
            'stats': {
                'Easy': stats_list(diffs.get('Easy',[])),
                'Hard': stats_list(diffs.get('Hard',[])),
                'All': stats_list(diffs.get('All',[]))
            }
        })
    
    # compute t-tests per difficulty
    from scipy.stats import ttest_ind # type: ignore
    
    def format_p_value(p):
        """Format p-value with scientific notation"""
        p_rounded = round(p, 2)
        if p > 0.05:
            return {'p_value': p_rounded, 'notation': 'N.S.'}
        elif p > 0.01:
            return {'p_value': p_rounded, 'notation': '*'}
        elif p > 0.001:
            return {'p_value': p_rounded, 'notation': '**'}
        else:
            return {'p_value': p_rounded, 'notation': '***'}
    
    ttest = {}
    model_keys = list(agg.keys())[:2]
    for diff in ('Easy','Hard','All'):
        if len(model_keys)==2:
            x = agg[model_keys[0]][diff]
            y = agg[model_keys[1]][diff]
            if x and y:
                _, p = ttest_ind(x, y, equal_var=False)
                p_formatted = format_p_value(p)
                ttest[diff] = {'N': min(len(x), len(y)), **p_formatted}
    return jsonify({'models': results, 'ttest': ttest})

@app.route('/speaker_talks', methods=['GET'])
@login_required
def speaker_talks():
    """Return list of all talk filenames seen in requests_questions.log"""
    talks = set()
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if 'file' in e:
                        talks.add(e['file'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify({'talks': []})
    return jsonify({'talks': sorted(talks)})

@app.route('/speaker_questions', methods=['GET'])
@login_required
def speaker_questions():
    """
    Return up to `num` random questions (with >=2 answers) for the given talk.
    Query params: talk=<filename>, num=<int>
    """
    talk = request.args.get('talk', '')
    try:
        num = max(0, int(request.args.get('num', '0')))
    except ValueError:
        num = 0

    # collect question entries for this talk
    qs = []
    try:
        with open('requests_questions.log', 'r') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get('file') == talk and 'prompt' in e:
                        qs.append(e)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify({'questions': []})

    # attach answers from Redis, only keep those with >=2
    results = []
    for q in qs:
        rid = q['request_id']
        pattern = f"result:{rid}:*"
        m_ans = {}
        for key in redis_client.scan_iter(match=pattern):
            data = redis_client.get(key)
            if not data: continue
            r = json.loads(data)
            m_ans[r['model']] = r['summary']
        if len(m_ans) >= 2:
            results.append({
                'request_id': rid,
                'prompt': q.get('prompt'),
                'file': q.get('file'),
                'model_answers': m_ans
            })

    # random subset
    if num > 0 and results:
        results = random.sample(results, min(num, len(results)))
    return jsonify({'questions': results})

# load GitHub repo URL
GITHUB_REPO_URL = os.getenv('GITHUB_REPO_URL', 'https://github.com/your-org/your-repo.git')

# configure periodic clone every 60s
celery.conf.beat_schedule = {
    'periodic-repo-sync': {
        'task': 'app.update_repo',
        'schedule': 60.0
    }
}

@celery.task(name='app.update_repo')
def update_repo():
    tmpdir = tempfile.mkdtemp()
    # clone or pull latest
    subprocess.run(['git', 'clone', GITHUB_REPO_URL, tmpdir], check=True)
    dest = app.config['UPLOAD_FOLDER']
    # clear old uploads
    for name in os.listdir(dest):
        path = os.path.join(dest, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    # copy fresh contents
    for name in os.listdir(tmpdir):
        src = os.path.join(tmpdir, name)
        dst = os.path.join(dest, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    # record sync time
    redis_client.set('repo_last_update', datetime.now().isoformat())

@app.route('/repo_status', methods=['GET'])
@login_required
def repo_status():
    last = redis_client.get('repo_last_update')
    return jsonify({'last_update': last.decode() if last else None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)
