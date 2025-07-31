import os
import json
import logging
from dotenv import load_dotenv  #type: ignore

# load env and app context
load_dotenv()
from app import (
    app, 
    extract_text_from_pdf, 
    process_summary, 
    redis_client, 
    PENDING_QUESTIONS_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # fetch all pending questions
    pending = redis_client.lrange(PENDING_QUESTIONS_KEY, 0, -1)
    if not pending:
        logger.info("No pending questions found.")
        return

    for raw in pending:
        try:
            q = json.loads(raw)
            pdf = q.get("selected_pdf")
            req_id = q.get("request_id")
            prompt = q.get("prompt_prefix")
            nick = q.get("nickname")
            pdf_path = os.path.join(app.config['PDF_STORAGE_FOLDER'], pdf)

            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not yet available: {pdf}")
                continue

            text = extract_text_from_pdf(pdf_path)
            # try each model once; the task itself will rotate through your API keys
            for model in ['openai', 'perplexity']:
                try:
                    logger.info(f"Attempting {model} for pending {req_id}")
                    # run task synchronously (blocking); you'll get an exception on failure
                    result = process_summary.apply(
                        args=(text, prompt, model, req_id, None, nick)
                    ).get(timeout=60)
                    logger.info(f"✅ Success for {req_id} on {model}: removed from pending")
                    # remove exactly one matching entry from the list
                    redis_client.lrem(PENDING_QUESTIONS_KEY, 1, raw)
                    break
                except Exception as e:
                    logger.warning(f"{model} failed for {req_id}: {e}")
            else:
                logger.error(f"All models failed for pending {req_id}; leaving in queue")
        except Exception as e:
            logger.error(f"Failed to process pending entry: {e}")

if __name__ == "__main__":
    main()
