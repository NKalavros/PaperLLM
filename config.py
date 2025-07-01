import os
from werkzeug.security import generate_password_hash #type: ignore
from dotenv import load_dotenv #type: ignore

# Load environment variables from .env file
load_dotenv()

# User credentials from .env file or environment variables
USER_DATABASE = {
    'admin': {
        "password": generate_password_hash(os.getenv("ADMIN_PASSWORD", "adminpass")),
        "role": "admin"
    },
    os.getenv("GUSTAVO_USERNAME", "gustavo"): {
        "password": generate_password_hash(os.getenv("GUSTAVO_PASSWORD", "gustavopass")),
        "role": "gustavo"
    },
    os.getenv("AUDIENCE_USERNAME", "audience"): {
        "password": generate_password_hash(os.getenv("AUDIENCE_PASSWORD", "audiencepass")),
        "role": "audience"
    },
    os.getenv("SPEAKER_USERNAME", "speaker"): {
        "password": generate_password_hash(os.getenv("SPEAKER_PASSWORD", "speakerpass")),
        "role": "speaker"
    }
}
