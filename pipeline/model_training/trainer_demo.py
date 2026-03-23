import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Select device automatically
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

# Move model to CPU manually if CUDA is not available
if device == "cpu":
    model.to(device)

# Set model to training mode
model.train()

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

# Load batch input from JSON
data_path = Path(__file__).resolve().parent.parent / "reinforcement_algorithm" / "policy_update_batch.json"
print("Loading data from:", data_path)

with open(data_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

results = []
batch_loss = 0.0

# Clear gradients before backward
optimizer.zero_grad()

# Process each example in the batch
for ex in examples:
    prompt_text = ex["prompt_text"]
    topology_text = ex["topology_text"]
    advantage = ex["advantage"]

    # Build the full sequence and the prompt-only prefix
    full_text = f"User request:\n{prompt_text}\n\nGenerated topology:\n{topology_text}"
    prompt_only = f"User request:\n{prompt_text}\n\nGenerated topology:\n"

    # Tokenize both sequences
    full_inputs = tokenizer(full_text, return_tensors="pt")
    prompt_inputs = tokenizer(prompt_only, return_tensors="pt")

    full_ids = full_inputs.input_ids.to(device)
    prompt_ids = prompt_inputs.input_ids.to(device)

    # Forward pass through the model
    outputs = model(full_ids)
    logits = outputs.logits

    # Shift logits and labels for causal language modeling
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    # Convert logits to log-probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)

    # Extract the log-probability of the actual target tokens
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Keep only the response part (the topology), not the prompt part
    prompt_length = prompt_ids.shape[1]
    response_log_prob = token_log_probs[:, prompt_length - 1:].sum()

    # Compute policy-gradient-style loss for this example
    loss = -response_log_prob * advantage

    # Accumulate batch loss as a tensor
    batch_loss = batch_loss + loss

    result = {
        "constraint_id": ex["constraint_id"],
        "prompt_text": prompt_text,
        "topology_text": topology_text,
        "advantage": advantage,
        "log_prob": response_log_prob.item(),
        "loss": loss.item(),
    }

    results.append(result)

    print("\n==============================")
    print("Constraint ID:", ex["constraint_id"])
    print("Prompt:")
    print(prompt_text)
    print("\nGenerated topology:")
    print(topology_text)
    print("\nAdvantage:")
    print(advantage)
    print("\nResponse log probability:")
    print(response_log_prob.item())
    print("\nPolicy gradient style loss:")
    print(loss.item())

# Compute mean batch loss
if examples:
    batch_loss = batch_loss / len(examples)

# Backward pass on the full batch
batch_loss.backward()

# Update model parameters
optimizer.step()

print("\n==============================")
print("Batch mean loss:", batch_loss.item())

# Save detailed outputs
output_path = Path(__file__).parent / "trainer_demo_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Saved detailed results to: {output_path}")

# Save batch summary
batch_summary = {
    "num_examples": len(examples),
    "batch_mean_loss": batch_loss.item() if examples else 0.0,
}

summary_path = Path(__file__).parent / "trainer_demo_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(batch_summary, f, indent=2)

print(f"Saved batch summary to: {summary_path}")