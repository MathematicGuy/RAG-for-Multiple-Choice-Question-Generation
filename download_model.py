from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import snapshot_download


# Load tokenizer and model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,         # Use float16 for memory efficiency
    device_map="auto"                  # Automatically use GPU if available
)

# Sample prompt
prompt = "Explain what is machine learning in simple terms."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# Decode and print
print(tokenizer.decode(output[0], skip_special_tokens=True))

snapshot_download(repo_id="google/gemma-2b-it")
local_path = "/your/downloaded/path/google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path)
