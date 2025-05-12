import os
from werkzeug.security import generate_password_hash

# User credentials from environment variables
USER_DATABASE = {
    os.getenv("ADMIN_USERNAME", "admin"): {
        "password": generate_password_hash(os.getenv("ADMIN_PASSWORD", "changeme")),
        "role": "admin"
    },
    # Add more users as needed
}
