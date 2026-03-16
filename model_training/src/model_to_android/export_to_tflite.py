import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name_or_path = "distilbert_model_android"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
model.eval()

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["logits"]  # only return the tensor

wrapped_model = WrappedModel(model)

# Dummy input: batch=1, seq_len=128
dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128))

# Trace and save
traced_model = torch.jit.trace(wrapped_model, dummy_input)
traced_model.save("distilbert_traced.pt")

print("✅ TorchScript model saved as distilbert_traced.pt")
