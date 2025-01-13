import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from src module
from src.data_preprocessing import preprocess_data
from transformers import BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Force CPU usage for training
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load dataset
dataset = load_dataset("samsum")

# Preprocess dataset
processed_dataset = dataset.map(preprocess_data, batched=True)

# Load the model (using smaller BART model for reduced memory usage)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/fine_tuned",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    per_device_train_batch_size=2,  # Reduced batch size for memory efficiency
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,  # Important for custom datasets
    no_cuda=True,  # Disable GPU, use CPU
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
)

# Train the model
if __name__ == "__main__":
    trainer.train()
