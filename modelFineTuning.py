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
#input_features  -> Audio "path" to be parametrized
#   must be handled by the feature extractor 
#labels -> transcription
#   the labels by the tokenizer.

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the processor just once here
tiny_model_directory = "./models/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(tiny_model_directory, language="English", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(tiny_model_directory, language="English", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(tiny_model_directory)


def load_and_prepare_data():
    logging.info("Starting to load and prepare data.")
    # Assume df is your original DataFrame
    # First, let's split the data into training and temp (which will later be divided into validation and test)
    # Now, let's split the temp data into validation and test sets
    dataFrame = pd.read_csv("metadata.csv")
    logging.info("Data loaded from metadata.csv.")

    train_df, temp_df = train_test_split(dataFrame, test_size=0.1, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    logging.info(f"Data split into train, validation, and test sets. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # Create Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})
    # Use the .map method to apply the function to all elements in the dataset
    dataset_dict = dataset_dict.map(prepare_features, remove_columns=["path"])
    dataset_dict = dataset_dict.map(prepare_labels, remove_columns=["transcription"])

    return dataset_dict
    #60-20-20
        #test_size=0.2
        #test_size=0.25

    #90-5-5
        #test_size=0.05
        #test_size=1/19

def initialize_model_and_optimization():
    logging.info("Initializing model and optimizer.")

    model = WhisperForConditionalGeneration.from_pretrained(tiny_model_directory)
    logging.info("Model loaded from pretrained directory.")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to("cuda")
    logging.info("Model configured and moved to CUDA.")


    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    logging.info("Optimizer initialized.")

    return model, optimizer

def training_loop(train_loader, model, optimizer):
    logging.info("***************Starting the training loop***************")
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        logging.info(f"*-*-*-*-*-*-*-*Processing training batch {batch_idx + 1} of {num_batches}.*-*-*-*-*-*-*-*")

        input_values = batch["input_values"].to("cuda")  # encoder inputs
        logging.info(f"Moved encoder inputs to CUDA. Shape: {input_values.shape}")

        labels = batch["labels"].to("cuda")  # ground truth for loss calculation
        logging.info(f"Moved labels to CUDA. Shape: {labels.shape}")

        # It's common to use the labels as decoder_input_ids. You shift the labels by one 
        # to predict the next token in sequence. Here is how you can do it:
        # Forming decoder input by shifting the labels
        decoder_input_ids = labels[:, :-1].contiguous()  # remove last token
        logging.info(f"Generated decoder input ids by shifting labels. Shape: {decoder_input_ids.shape}")

        lm_labels = labels[:, 1:].clone().detach()  # remove first token
        lm_labels[labels[:, 1:] == -100] = -100  # set padding tokens to -100
        logging.info(f"Generated labels for loss computation. Shape: {lm_labels.shape}")

        # Forward pass
        logging.info("Starting forward pass.")
        outputs = model(input_values, decoder_input_ids=decoder_input_ids, labels=lm_labels)
        loss = outputs.loss  # model outputs are always tuple in transformers (see doc)
        logging.info(f"Computed loss: {loss.item()}")

        # Backward pass
        logging.info("Starting backward pass.")
        loss.backward()
        logging.info("Completed backward pass.")

        #optimization steps
        logging.info("Starting optimization step.")
        optimizer.step()
        logging.info("Completed optimization step.")

        logging.info("Zeroing out optimizer gradients.")
        optimizer.zero_grad()
        logging.info("Zeroed out optimizer gradients.")

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logging.info(f"Training loop completed. Average loss: {avg_loss}")

    return avg_loss  # return last training loss

def evaluation_loop(val_loader, model):
    logging.info("-----------------Starting the evaluation loop-----------------")

    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            logging.info(f"*-*-*-*-*-*-*-*Evaluating batch {batch_idx + 1} of {num_batches}.*-*-*-*-*-*-*-*")

            input_values = batch["input_values"].to("cuda")
            logging.info(f"Moved encoder inputs to CUDA. Shape: {input_values.shape}")

            labels = batch["labels"].to("cuda")
            logging.info(f"Moved labels to CUDA. Shape: {labels.shape}")

            # Forming decoder input by shifting the labels
            decoder_input_ids = labels[:, :-1].contiguous()
            logging.info(f"Generated decoder input ids by shifting labels. Shape: {decoder_input_ids.shape}")

            # For loss computation, also shift labels and align with decoder's output
            lm_labels = labels[:, 1:].clone().detach()
            lm_labels[labels[:, 1:] == -100] = -100
            logging.info(f"Generated labels for loss computation. Shape: {lm_labels.shape}")

            logging.info("Starting forward pass.")
            outputs = model(input_values, decoder_input_ids=decoder_input_ids, labels=lm_labels)
            loss = outputs.loss  # Model outputs are always tuple in transformers (see doc)
            logging.info(f"Computed loss: {loss.item()}")

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    logging.info(f"Evaluation loop completed. Average loss: {avg_loss}")

    return avg_loss  # return average evaluation loss


def modelFineTuning():
    logging.info("Starting model fine-tuning.")
    # Step 1: Load and Prepare Data
    logging.info("Step 1: Loading and preparing data.")
    dataset_dict = load_and_prepare_data()

    # Step 2: Initialize Model and Optimizer
    logging.info("Step 2: Initializing model and optimizer.")
    model, optimizer = initialize_model_and_optimization()

    # Step 3: Create DataLoaders
    logging.info("Step 3: Creating DataLoaders.")
    train_loader = DataLoader(dataset_dict['train'], batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_dict['validation'], batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_dict['test'], batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Step 4: Initialize Criterion
    criterion = torch.nn.CTCLoss().to("cuda")

    # Initialize best_loss to a high value
    best_loss = float('inf')

    #training
    # Step 1: Assuming dataset_dict contains train, test, and validation sets
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['validation']
    test_dataset = dataset_dict['test']

    # Step 2: Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Step 3: Initialize model

    # Step 4: Define the optimizer and loss function
    criterion = torch.nn.CTCLoss()
    # Initialize best_loss to a high value
    best_loss = float('inf')

    # Step 5: Training and Evaluation Loops
    for epoch in range(10):
        print(f"*-*-*-*-*-*-*-*-*Starting Epoch: {epoch + 1}*-*-*-*-*-*-*-*-*")

        train_loss = training_loop(train_loader, model, optimizer)
        val_loss = evaluation_loop(val_loader, model)

        # Save checkpoints
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"best_model_checkpoint_epoch{epoch}.pth")

        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Save the final model
    torch.save(model.state_dict(), "final_model_checkpoint.pth")

    # Compute WER
    avg_wer = compute_wer(test_loader, model)
    logging.info(f"Final Best Validation Loss: {best_loss}, Final WER: {avg_wer}")

    return {'best_val_loss': best_loss, 'avg_wer': avg_wer}, model  # Return metrics and model for further analysis


def prepare_features(batch):
    logging.info("Starting the prepare_features function.")

    try:
        waveform, sample_rate = torchaudio.load(batch["path"], normalize=True)
        logging.info(f"Original sample rate: {sample_rate}")

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            sample_rate = 16000

        processed = processor(audio=waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
        input_features = processed['input_features']

        if input_features is not None:
            logging.info(f"input_features shape before: {input_features.shape}")
        else:
            logging.error("input_features is None")
            return batch

        batch["input_values"] = input_features
        logging.info(f"Input values shape after: {input_features.shape}")

    except Exception as e:
        logging.error(f"An error occurred in prepare_features: {e}")
        return batch
    
    return batch

def prepare_labels(batch):
    logging.info("Starting the prepare_labels function.")
    try:
        labels = processor(text=batch["transcription"])["input_ids"]
        logging.info(f"Labels shape: {len(labels)}")

        batch["labels"] = labels

    except Exception as e:
        logging.error(f"An error occurred in prepare_labels: {e}")
        return batch

    return batch

def collate_fn(batch):
    logging.info("Starting the collate function.")
    # Convert lists to tensors if they are not
    for i, item in enumerate(batch):
        if not isinstance(item.get('input_values', None), torch.Tensor):
            item['input_values'] = torch.tensor(item['input_values'])

    # Make sure that the tensors are 3D: [batch, features, time]
    # If your tensor inside `input_values` is not 3D, reshape it to make it 3D.
    input_values = [x['input_values'].squeeze(0) for x in batch]  # Remove the batch dimension, if it's 1.
    
    # Pad sequence input_values
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    logging.info(f"Input values padded shape: {input_values_padded.shape}")

    # Padding labels
    labels = [torch.tensor(x['labels']) for x in batch]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    logging.info(f"Labels padded shape: {labels_padded.shape}")

    return {'input_values': input_values_padded, 'labels': labels_padded}

def compute_wer(data_loader, model):
    logging.info("Starting the WER (Word Error Rate) computation.")
    model.eval()
    avg_wer = 0
    total_wer = 0  # Initialize this
    total_samples = 0  # Initialize this too

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            logging.info(f"Computing WER for batch {batch_idx + 1}.")

            input_values = batch["input_values"].to("cuda")
            logging.info(f"Moved encoder inputs to CUDA. Shape: {input_values.shape}")

            labels = batch["labels"].to("cuda")
            logging.info(f"Moved labels to CUDA. Shape: {labels.shape}")

            decoder_input_ids = labels[:, :-1].contiguous()  # Assuming the labels have to be shifted similarly for WER computation
            logging.info(f"Generated decoder input ids by shifting labels. Shape: {decoder_input_ids.shape}")

            logging.info("Starting forward pass.")
            outputs = model(input_values, decoder_input_ids=decoder_input_ids)  # Update this line to match how you use the model in this method
            
            logits = outputs.logits

            decoded_preds = tokenizer.batch_decode(logits.argmax(dim=-1))
            decoded_labels = tokenizer.batch_decode(labels)

            for ref, hyp in zip(decoded_labels, decoded_preds):
                total_wer += wer(ref, hyp)
                total_samples += 1
                logging.info(f"Total WER so far: {total_wer}")
                logging.info(f"Total samples so far: {total_samples}")
                avg_wer = total_wer / total_samples
                print(f"Average WER on the test set: {avg_wer:.4f}")

    return avg_wer

