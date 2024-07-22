# import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset
import os
from github import Github

#MPS error
# torch.backends.mps.enabled = False

# GitHub setup
github_token = "YOUR_TOKEN"
repo_name = "ShaliniAnandaPhD/Sea_Sifter"

# file_path = "train.py"
file_path = "advanced_microplastic_forecasting.py" #MH added

# Initialize GitHub client
g = Github(github_token)

# Get the repository
repo = g.get_repo(repo_name)

# Get the file content
file_content = repo.get_contents(file_path).decoded_content.decode()

# print(file_content)

# for content_file in contents:
#     print(content_file)

# Save the content to a local file
with open("local_train.py", "w") as f:
    f.write(file_content)

# Load the dataset
# dataset = load_dataset("text", data_files={"train": ["local_train.py"]})
# dataset = load_dataset("lhoestq/demo1") #MH added
dataset = load_dataset("json", data_files="test_data.json") #MH added

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments

# MPS Issues: https://discuss.huggingface.co/t/runtimeerror-placeholder-storage-has-not-been-allocated-on-mps-device/42999/2

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    # per_device_train_batch_size=8,
    per_device_train_batch_size=4096,
    save_steps=10_000,
    save_total_limit=2,
    no_cuda=True, #MH added to resolve MPS
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_codebert")
tokenizer.save_pretrained("./fine_tuned_codebert")
