from werkzeug.security import generate_password_hash

# User credentials
USER_DATABASE = {
    "gustavo": {
        "password": generate_password_hash("v6SRh17Avus2bTct"),
        "role": "admin"
    },
    # Add more users as needed
}