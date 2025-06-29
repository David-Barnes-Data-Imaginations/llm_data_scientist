from transformers import AutoModelForSequenceClassification

# Load from safetensors
model = AutoModelForSequenceClassification.from_pretrained(
    ".", trust_remote_code=True
)

# Save in PyTorch format
model.save_pretrained("./models/DeepSeek-R1-0528-Qwen3-8B/1", safe_serialization=False)