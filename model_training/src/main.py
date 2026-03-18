import argparse
from huggingface_hub import login

from utils import load_config
from data import load_dataset_from_config, tokenize_dataset
from models import build_model, build_tokenizer
from train import train

def main(config_path: str):

    config = load_config(config_path)

    login(token=config["hf_token"])

    dataset = load_dataset_from_config(config["dataset"])

    for model_name in config["models"]:

        tokenizer = build_tokenizer(model_name)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)

        model = build_model(model_name, num_labels=2)

        trainer = train(
            model,
            tokenized_dataset,
            tokenizer,
            config["training"],
            model_name.replace('/', '_'),
            config["hf_train_version"],
            output_dir=f"output/{model_name.replace('/', '_')}" + config["hf_train_version"],
        )

        trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    main(args.config)