---
library_name: transformers
base_model: huawei-noah/TinyBERT_General_4L_312D
tags:
- generated_from_trainer
model-index:
- name: huawei-noah_TinyBERT_General_4L_312D_train_v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# huawei-noah_TinyBERT_General_4L_312D_train_v2

This model is a fine-tuned version of [huawei-noah/TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1359

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step  | Validation Loss |
|:-------------:|:-----:|:-----:|:---------------:|
| 0.1832        | 1.0   | 16723 | 0.1427          |
| 0.1115        | 2.0   | 33446 | 0.1392          |
| 0.1118        | 3.0   | 50169 | 0.1359          |


### Framework versions

- Transformers 5.3.0
- Pytorch 2.10.0+cu128
- Datasets 4.6.1
- Tokenizers 0.22.2
