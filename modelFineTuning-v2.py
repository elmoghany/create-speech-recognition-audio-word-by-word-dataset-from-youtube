import logging
import torchaudio
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from jiwer import wer
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from transformers import TrainerCallback


import pydub
import numpy as np
import os
import csv
import json
from pydub.utils import mediainfo
from pydub import AudioSegment
#input_features  -> Audio "path" to be parametrized
#   must be handled by the feature extractor 
#labels -> transcription
#   the labels by the tokenizer.

logging.basicConfig(level=logging.INFO)

# Initialize an empty list to hold the new data dictionaries
new_data_dicts = []

# Read existing CSV file
# with open('metadata.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
    
#     # Loop through each row in the CSV
#     for row in csv_reader:
#         print("sampling & creating dataset row: ", row)
#         path = row["path"]
#         transcription = row["transcription"]
        
#         # Read audio data using pydub
#         audio = AudioSegment.from_mp3(path)
        
#         # Change the sample rate
#         new_sampling_rate = 16000  # specify the new sampling rate
#         resampled_audio = audio.set_frame_rate(new_sampling_rate)
        
#         # Get the raw audio data as an array
#         audio_data = np.array(resampled_audio.get_array_of_samples())

#         # Normalize the array to float32 in the range [-1, 1]
#         audio_data = audio_data.astype('float32') / 32767.0  # Since the data is 16-bit PCM

#         # Create the new data dictionary
#         data_dict = {
#             "path": path,
#             "transcription": transcription,
#             "audio": {
#                 "path": path,
#                 "sampling_rate": new_sampling_rate,
#                 "array": audio_data.tolist(),
#                     # "dtype": "float32"
#             }
#         }
        
#         # Append the new data dictionary to the list
#         new_data_dicts.append(data_dict)
# print(new_data_dicts[:1])
        
# # Save new data to a JSON file
# with open('new_data.json', 'w') as json_file:
#     print("new_data.json")
#     json.dump(new_data_dicts, json_file)

# Read existing JSON file
with open('new_data.json', 'r') as json_file:
    new_data_dicts = json.load(json_file)
    
    # Loop through each entry in the list of dictionaries
    for data_dict in new_data_dicts:
        print("sampling & creating dataset row:")
        path = data_dict["path"]
        transcription = data_dict["transcription"]
        
        # Here, audio information is already available in data_dict["audio"]
        new_sampling_rate = data_dict["audio"]["sampling_rate"]
        audio_data = np.array(data_dict["audio"]["array"])
print(new_data_dicts[:1])

# Initialize the processor just once here
from transformers import WhisperForConditionalGeneration

tiny_model_directory = "./models/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(tiny_model_directory, language="English", task="transcribe", chunk_length = 10)
model = WhisperForConditionalGeneration.from_pretrained(tiny_model_directory)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.to("cuda")

logging.info("Starting to load and prepare data.")
df = pd.json_normalize(new_data_dicts)
logging.info("Data converted to Pandas DataFrame.")

train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


logging.info(f"Data split into train, validation, and test sets. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

from collections import Counter
######
######
######
#####Plot top 20 most repeated words
# Assuming dataset_dict["train"]["transcription"] contains all the transcription texts
print(dataset_dict["train"]["transcription"][:5])
all_transcriptions = dataset_dict["train"]["transcription"]

# Create one big list of words from all transcriptions
all_transcriptions_tuple = tuple(all_transcriptions)
print(all_transcriptions_tuple[:5])

# Count the occurrences of each word
word_counter = Counter(all_transcriptions_tuple)

# Get the top 30 most common words and their counts
most_common_30 = word_counter.most_common(30)

# Separate the words and their counts into two lists
words, counts = zip(*most_common_30)

# Text output
print("Top 30 most common words:")
for word, count in most_common_30:
    print(f"{word}: {count}")

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Count')
plt.ylabel('Word')
plt.title('Top 20 Most Common Words')
plt.gca().invert_yaxis()  # Reverse the order for better readability
plt.show()

###########
###########
###########
###########

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio_array = batch['audio.array']
    audio_sampling_rate = batch['audio.sampling_rate']

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor_class(audio_array, sampling_rate=audio_sampling_rate).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch


print("Columns in train set:", dataset_dict["train"].column_names)
dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"])


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

from sklearn.metrics import confusion_matrix
import numpy as np

true_labels = []
predicted_labels = []

def compute_metrics(pred):
    global true_labels, predicted_labels  # Declare as global to modify

    # For WER
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    # For confusion matrix
    pred_ids_cm = pred.predictions.argmax(-1)
    
    print(pred_ids_cm.shape)
    print(label_ids.shape)

    # Flatten and filter out padding tokens
    pred_ids_cm = pred_ids_cm[label_ids != processor.tokenizer.pad_token_id]
    label_ids_cm = label_ids[label_ids != processor.tokenizer.pad_token_id]

    # Extend the global lists
    true_labels.extend(label_ids_cm.tolist())
    predicted_labels.extend(pred_ids_cm.tolist())

    # You can compute the confusion matrix here or later
    cm = confusion_matrix(true_labels, predicted_labels)

    return {"wer": wer, "confusion matrix: ": cm}

####
####
####
####
#Loss & Accuracy curve
#Learning curve
# Define a custom callback to save logs
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_loss_set = []
        self.eval_loss_set = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs.keys():  # Training loss
            self.train_loss_set.append(logs["loss"])
            self.steps.append(state.global_step)
        if "eval_loss" in logs.keys():  # Eval loss
            self.eval_loss_set.append(logs["eval_loss"])

# Initialize callback
metrics_callback = MetricsCallback()

####
####
####
####

# 4. To store gradient norms
gradient_norms = []

class GradientNormsCallback(TrainerCallback):
    def on_backward_end(self, args, state, control, model=None, **kwargs):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradient_norms.append(total_norm)

# Initialize gradient norms callback
gradient_norms_callback = GradientNormsCallback()

####
####
####
####
# 5 & 6. To store precision-recall and ROC values
y_true = []
y_pred = []

class EvalPredictionCallback(TrainerCallback):
    def on_prediction_end(self, args, state, control, predictions, label_ids, metrics, **kwargs):
        # Assuming a binary classification problem; 
        # you may need to adjust for your specific case
        preds = torch.softmax(predictions.predictions, dim=-1)[:, 1]
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(label_ids.cpu().numpy())

# Initialize eval callback
eval_pred_callback = EvalPredictionCallback()

############
############
############
############
############
# Function to plot ATTENTION
# def plot_attention(attention, input_tokens, output_tokens):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sns.heatmap(attention, ax=ax, cmap='viridis')

#     ax.set_xticks(range(len(input_tokens)))
#     ax.set_yticks(range(len(output_tokens)))

#     ax.set_xticklabels(input_tokens)
#     ax.set_yticklabels(output_tokens)

#     plt.show()

# model.eval()
# # Assuming 'dataset_dict["test"]' is your test dataset
# test_dataloader = torch.utils.data.DataLoader(dataset_dict["test"], batch_size=1, shuffle=True)

# # Take one batch from test dataset
# for i, batch in enumerate(test_dataloader):
#     input_features = batch["input_features"]
#     labels = batch["labels"]
    
#     # Convert to tensors if they are not
#     if not isinstance(input_features, torch.Tensor):
#         input_features = torch.tensor(input_features)
#     if not isinstance(labels, torch.Tensor):
#         labels = torch.tensor(labels)

#     print("Type:", type(input_features), type(labels))
#     print("Shape:", input_features.shape, labels.shape)

#     # Forward pass
#     with torch.no_grad():
#         outputs = model(input_features=input_features, labels=labels, output_attentions=True)

#     # 'outputs' now contains attention weights
#     # Plotting attention for the first head of the first layer for the first example in the batch
#     attention = outputs.attentions[0][0, 0, :, :].cpu().detach().numpy()

#     # Decode tokens to text
#     input_tokens = processor.feature_extractor.batch_decode(input_features.cpu())
#     output_tokens = processor.tokenizer.batch_decode(labels.cpu())

#     plot_attention(attention, input_tokens[0].split(), output_tokens[0].split())

#     # Plot only the first batch for demonstration
#     break

###########
###########
###########
###########

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-1st-training-test",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[metrics_callback, gradient_norms_callback, eval_pred_callback]
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

#######
#######
#######
#######
#Confusion Matrix: Useful for classification tasks to understand the type of errors your model is making.
# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#######
#######
#######
#######

#Loss Curve: Plotting the training and validation loss helps you understand how well the model is learning. 
# You want the loss to decrease over time without the validation loss increasing (which could indicate overfitting)

#Accuracy Curve: If applicable, plotting training and validation accuracy can provide insights into the model's performance. 
# However, for tasks like machine translation or text-to-speech, specialized metrics might be more useful.
#Loss & Accuracy curve
# Now plot the saved metrics
plt.figure()
plt.plot(metrics_callback.steps, metrics_callback.train_loss_set, label="Training Loss")
plt.plot(metrics_callback.steps, metrics_callback.eval_loss_set, label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

#######
#######
#######
#######
#Learning Rate: If you're using a learning rate schedule, plotting the learning rate over time can be useful.
#Learning curve
# This will serve as a makeshift learning curve
plt.figure()
plt.plot(metrics_callback.steps, metrics_callback.train_loss_set, label="Training Loss")
plt.title("Learning Curve")
plt.xlabel("Steps")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

#######
#######
#######
#######
#Gradient Norms: Monitoring the norms of gradients can help in diagnosing issues related to 
# vanishing or exploding gradients.
# 4. Plot gradient norms
plt.figure()
plt.plot(range(len(gradient_norms)), gradient_norms)
plt.title('Gradient Norms')
plt.xlabel('Training Steps')
plt.ylabel('Gradient Norm')
plt.show()

#######
#######
#######
#######
#Precision-Recall Curve: Especially useful for imbalanced datasets, 
# this curve helps you understand the trade-off between precision and recall for your model.
# 5. Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)
plt.figure()
plt.plot(recall, precision)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

#######
#######
#######
#######
#ROC Curve and AUC: Receiver operating characteristic curve is a graphical plot that 
# illustrates the diagnostic ability of a binary classifier.
# 6. Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#######
#######
#######
#######