with open("hf_env", 'r') as hf_token:
    user_token = hf_token.read().strip()

from huggingface_hub import login
login(token=user_token)
