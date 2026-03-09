from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

def train(model, tokenized_dataset, tokenizer, training_config, model_name, train_version, output_dir="fine-tuned-models"):
    # Data collator to dynamically pad sequences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define Hugging Face training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
        hub_model_id="OliverHeine/" + model_name + train_version
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    return trainer