import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration

# Load model and tokenizer
model = BartForConditionalGeneration.from_pretrained("models/fine_tuned/checkpoint-5523")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Load ROUGE metric from evaluate library
rouge = evaluate.load("rouge")

def evaluate_model(dialogue, reference_summary):
    """
    Evaluates the model's summarization quality using ROUGE.
    """
    # Tokenize input dialogue
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=64, min_length=10, do_sample=False)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Compute ROUGE metric
    metrics = rouge.compute(predictions=[generated_summary], references=[reference_summary])
    return generated_summary, metrics

# Example usage
if __name__ == "__main__":
    dialogue = "Hey, are you coming to the meeting tomorrow? It's at 10 AM."
    reference_summary = "Meeting scheduled at 10 AM."
    
    generated_summary, metrics = evaluate_model(dialogue, reference_summary)
    print("Generated Summary:", generated_summary)
    print("ROUGE Metrics:", metrics)
