import inspect

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

def train(model, tokenized_dataset, tokenizer, training_config, model_name, train_version, output_dir="fine-tuned-models"):
    # Data collator to dynamically pad sequences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define Hugging Face training arguments
    training_args_kwargs = dict(
        output_dir=output_dir,
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

    # transformers>=5 uses `processing_class`, older versions use `tokenizer`
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    return trainer