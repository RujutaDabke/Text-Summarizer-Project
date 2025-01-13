from transformers import BartTokenizer

# Initialize tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def preprocess_data(example):
    """
    Preprocesses the dialogue and summary for model input.
    """
    inputs = tokenizer(
        example["dialogue"], max_length=512, truncation=True, padding="max_length"
    )
    outputs = tokenizer(
        example["summary"], max_length=64, truncation=True, padding="max_length"
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
    }
