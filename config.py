import os
from werkzeug.security import generate_password_hash #type: ignore
from dotenv import load_dotenv #type: ignore

# Load environment variables from .env file
load_dotenv()

# User credentials from .env file or environment variables
USER_DATABASE = {
    os.getenv("ADMIN_USERNAME", "admin"): {
        "password": generate_password_hash(os.getenv("ADMIN_PASSWORD", "changeme")),
        "role": "admin"
    },
    # Add more users as needed
}
