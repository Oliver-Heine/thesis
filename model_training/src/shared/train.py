import inspect

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train(model, tokenized_dataset, tokenizer, training_config, model_name, output_dir="fine-tuned-models"):
    # Data collator to dynamically pad sequences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define Hugging Face training arguments
    training_args_kwargs = dict(
        output_dir=output_dir,
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=4,  # or 8 on Windows
        dataloader_pin_memory = True,
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        num_train_epochs=training_config["epochs"],
        weight_decay=training_config.get("weight_decay", 0.0),
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        seed=training_config.get("seed", 42),
        push_to_hub=True,
        hub_model_id="OliverHeine/" + model_name
    )

    # transformers>=5 uses `eval_strategy`, older versions use `evaluation_strategy`
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    # Initialize Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    )

    # Add compute_metrics here
    trainer_kwargs["compute_metrics"] = compute_metrics

    # transformers>=5 uses `processing_class`, older versions use `tokenizer`
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    return trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }