import os

# Tests never call the real LLM
os.environ["USE_LLM"] = "0"