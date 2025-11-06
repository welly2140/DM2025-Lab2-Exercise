import os
from dotenv import load_dotenv
env_path = "./config/.env"
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)