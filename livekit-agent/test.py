import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import random
import numpy as np

# Set a random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1️⃣ Load Dataset
dataset = load_dataset("iitm-ddp/iiith-indic-speech")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 2️⃣ Prepare Labels
label_list = sorted(set(example["emotion"] for example in dataset["train"]))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def preprocess(example):
    audio = example["audio"]
    example["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    example["labels"] = label2id[example["emotion"]]
    return example

# 3️⃣ Load Pretrained Wav2Vec2
checkpoint = "facebook/wav2vec2-large-robust"
processor = Wav2Vec2Processor.from_pretrained(checkpoint)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label,
)

# 4️⃣ Preprocess
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names, num_proc=4)

# 5️⃣ Define Trainer
args = TrainingArguments(
    output_dir="./wav2vec2-emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.005,
    logging_steps=50,
    report_to="none",
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
)

def compute_metrics(eval_pred):
    import evaluate
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

# 6️⃣ Train
trainer.train()

# 7️⃣ Save
trainer.save_model("final_emotion_model")
processor.save_pretrained("final_emotion_model")
