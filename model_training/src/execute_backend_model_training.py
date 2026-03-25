import argparse
from huggingface_hub import login

from shared.utils import load_config
from shared.backend_data_loader import load_dataset_from_config, tokenize_dataset
from shared.models import build_tokenizer_backend, build_model_backend
from shared.train import train

def main(config_path: str):

    config = load_config(config_path)

    login(token=config["hf_token"])

    dataset = load_dataset_from_config(config["dataset_backend"])

    for model_name in config["models_backend"]:

        tokenizer = build_tokenizer_backend(model_name)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, config["training_backend"]["max_length"])

        model = build_model_backend(model_name, num_labels=2, tokenizer=tokenizer)

        trainer = train(
            model,
            tokenized_dataset,
            tokenizer,
            config["training_backend"],
            model_name.replace('/', '_'),
            config["hf_backend_train_version"],
            output_dir=f"output/{model_name.replace('/', '_')}" + config["hf_backend_train_version"],
        )

        trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    main(args.config)