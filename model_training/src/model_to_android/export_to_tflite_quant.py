import torch
from transformers import AutoModelForSequenceClassification

model_name = "OliverHeine/distilbert-base-uncased_train_v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # only quantize Linear layers
    dtype=torch.qint8
)

# Trace for mobile
dummy_input_ids = torch.randint(0, 100, (1, 128))
dummy_attention_mask = torch.ones(1, 128)
traced_model = torch.jit.trace(quantized_model, (dummy_input_ids, dummy_attention_mask))

# Save
traced_model.save("distilbert_traced_quantized.pt")
print("Quantized TorchScript model saved (Android-ready)")
