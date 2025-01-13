from transformers import BartTokenizer, BartForConditionalGeneration

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained("D:/CS 584 -NLP/NLP Project/Text-Summarization-NLP-Project/models/fine_tuned/checkpoint-5523")
#model = BartForConditionalGeneration.from_pretrained("models/fine_tuned/checkpoint-5523")


def summarize_text(dialogue, max_length=64, min_length=10, do_sample=False):
    """
    Generates a summary for the given dialogue.
    Args:
        dialogue (str): The input conversational text.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
        do_sample (bool): Whether to use sampling for summary generation.
    Returns:
        str: The generated summary.
    """
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"], max_length=max_length, min_length=min_length, do_sample=do_sample
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
