# coding=utf-8

# from: https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.1-%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.ipynb

# !pip install transformers datasets

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "cola"
# model_checkpoint = "distilbert-base-uncased"
model_checkpoint = "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased"
cache_dir_datasets = '/data0/nfs_data/zhaoxi9/.cache/huggingface/datasets'
cache_dir_model = r'/data0/nfs_data/zhaoxi9/.cache/huggingface/transformers'
batch_size = 16

from datasets import load_dataset, load_metric

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task, cache_dir=cache_dir_datasets)
metric = load_metric('glue', actual_task)

print(dataset["train"][:5])

import datasets
import random
import pandas as pd
# from IPython.display import display, HTML


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    # display(HTML(df.to_html()))


# show_random_elements(dataset["train"])


import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
print(metric.compute(predictions=fake_preds, references=fake_labels))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir_model)

print(tokenizer("Hello, this one sentence!", "And this sentence goes with it."))

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


print(preprocess_function(dataset['train'][:5]))

encoded_dataset = dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, cache_dir=cache_dir_model)

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    "test-glue",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()


# ! pip install optuna
# ! pip install ray[tune]

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, cache_dir=cache_dir_model)


trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

print(best_run)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()



# (py37tch111) [zhaoxi9@localhost huggingface_transformers_tutorials]$ python -u text_classification.py
# Using the latest cached version of the module from /usr/home/zhaoxi9/.cache/huggingface/modules/datasets_modules/datasets/glue/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad (last modified on Fri May  6 17:08:23 2022) since it couldn't be found locally at glue., or remotely on the Hugging Face Hub.
# Reusing dataset glue (/data0/nfs_data/zhaoxi9/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 201.45it/s]
# Using the latest cached version of the module from /usr/home/zhaoxi9/.cache/huggingface/modules/datasets_modules/metrics/glue/91f3cfc5498873918ecf119dbf806fb10815786c84f41b85a5d3c47c1519b343 (last modified on Fri May  6 17:11:22 2022) since it couldn't be found locally at glue, or remotely on the Hugging Face Hub.
# {'sentence': ["Our friends won't buy this analysis, let alone the next one we propose.", "One more pseudo generalization and I'm giving up.", "One more pseudo generalization or I'm giving up.", 'The more we study verbs, the crazier they get.', 'Day by day the facts are getting murkier.'], 'label': [1, 1, 1, 1, 1], 'idx': [0, 1, 2, 3, 4]}
# {'matthews_correlation': 0.1972421118046462}
# {'input_ids': [101, 7592, 1010, 2023, 2028, 6251, 999, 102, 1998, 2023, 6251, 3632, 2007, 2009, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# Sentence: Our friends won't buy this analysis, let alone the next one we propose.
# Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
# {'input_ids': [[101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], [101, 2028, 2062, 18404, 2236, 3989, 1998, 1045, 1005, 1049, 3228, 2039, 1012, 102], [101, 2028, 2062, 18404, 2236, 3989, 2030, 1045, 1005, 1049, 3228, 2039, 1012, 102], [101, 1996, 2062, 2057, 2817, 16025, 1010, 1996, 13675, 16103, 2121, 2027, 2131, 1012, 102], [101, 2154, 2011, 2154, 1996, 8866, 2024, 2893, 14163, 8024, 3771, 1012, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
# Loading cached processed dataset at /data0/nfs_data/zhaoxi9/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-43fe1d12ddad841d.arrow
# Loading cached processed dataset at /data0/nfs_data/zhaoxi9/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-0df9e9a6f54f5b2e.arrow
# Loading cached processed dataset at /data0/nfs_data/zhaoxi9/.cache/huggingface/datasets/glue/cola/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-79668ff0df7e4fd3.arrow
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 5
#   Instantaneous batch size per device = 16
#   Total train batch size (w. parallel, distributed & accumulation) = 64
#   Gradient Accumulation steps = 1
#   Total optimization steps = 670
#   0%|                                                                                                                                                                                  | 0/670 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  20%|█████████████████████████████████▌                                                                                                                                      | 134/670 [01:29<04:29,  1.99it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.4958183169364929, 'eval_matthews_correlation': 0.4127412863004641, 'eval_runtime': 3.2803, 'eval_samples_per_second': 317.961, 'eval_steps_per_second': 5.182, 'epoch': 1.0}
#  20%|█████████████████████████████████▌                                                                                                                                      | 134/670 [01:32<04:29,  1.99it/s]Saving model checkpoint to test-glue/checkpoint-134
# Configuration saved in test-glue/checkpoint-134/config.json
# Model weights saved in test-glue/checkpoint-134/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-134/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-134/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  40%|███████████████████████████████████████████████████████████████████▏                                                                                                    | 268/670 [02:52<03:16,  2.04it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.49493351578712463, 'eval_matthews_correlation': 0.472806497515492, 'eval_runtime': 3.3721, 'eval_samples_per_second': 309.306, 'eval_steps_per_second': 5.041, 'epoch': 2.0}
#  40%|███████████████████████████████████████████████████████████████████▏                                                                                                    | 268/670 [02:56<03:16,  2.04it/s]Saving model checkpoint to test-glue/checkpoint-268
# Configuration saved in test-glue/checkpoint-268/config.json
# Model weights saved in test-glue/checkpoint-268/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-268/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-268/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                   | 402/670 [04:15<02:04,  2.15it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.4881631135940552, 'eval_matthews_correlation': 0.5274328033323212, 'eval_runtime': 3.3089, 'eval_samples_per_second': 315.213, 'eval_steps_per_second': 5.138, 'epoch': 3.0}
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                   | 402/670 [04:18<02:04,  2.15it/s]Saving model checkpoint to test-glue/checkpoint-402
# Configuration saved in test-glue/checkpoint-402/config.json
# Model weights saved in test-glue/checkpoint-402/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-402/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-402/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.3767, 'learning_rate': 5.074626865671642e-06, 'epoch': 3.73}
#  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                 | 536/670 [05:38<01:06,  2.01it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5451550483703613, 'eval_matthews_correlation': 0.5165560626605169, 'eval_runtime': 3.4881, 'eval_samples_per_second': 299.016, 'eval_steps_per_second': 4.874, 'epoch': 4.0}
#  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                 | 536/670 [05:41<01:06,  2.01it/s]Saving model checkpoint to test-glue/checkpoint-536
# Configuration saved in test-glue/checkpoint-536/config.json
# Model weights saved in test-glue/checkpoint-536/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-536/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-536/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 670/670 [07:01<00:00,  1.90it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.558754563331604, 'eval_matthews_correlation': 0.5340667882909217, 'eval_runtime': 3.4715, 'eval_samples_per_second': 300.447, 'eval_steps_per_second': 4.897, 'epoch': 5.0}
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 670/670 [07:05<00:00,  1.90it/s]Saving model checkpoint to test-glue/checkpoint-670
# Configuration saved in test-glue/checkpoint-670/config.json
# Model weights saved in test-glue/checkpoint-670/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-670/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-670/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/checkpoint-670 (score: 0.5340667882909217).
# {'train_runtime': 435.6633, 'train_samples_per_second': 98.138, 'train_steps_per_second': 1.538, 'train_loss': 0.3319309348490701, 'epoch': 5.0}
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 670/670 [07:15<00:00,  1.54it/s]
# The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17/17 [00:03<00:00,  5.26it/s]
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# [I 2022-05-06 17:37:36,789] A new study created in memory with name: no-name-3305eda0-4662-40bb-9586-f288e41c2d43
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 3
#   Instantaneous batch size per device = 4
#   Total train batch size (w. parallel, distributed & accumulation) = 16
#   Gradient Accumulation steps = 1
#   Total optimization steps = 1605
#   0%|                                                                                                                                                                                 | 0/1605 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.5253, 'learning_rate': 1.2608661800195405e-05, 'epoch': 0.93}
#  33%|███████████████████████████████████████████████████████▋                                                                                                               | 535/1605 [03:30<07:27,  2.39it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.47706958651542664, 'eval_matthews_correlation': 0.4552672986252464, 'eval_runtime': 3.3126, 'eval_samples_per_second': 314.856, 'eval_steps_per_second': 5.132, 'epoch': 1.0}
#  33%|███████████████████████████████████████████████████████▋                                                                                                               | 535/1605 [03:34<07:27,  2.39it/s]Saving model checkpoint to test-glue/run-0/checkpoint-535
# Configuration saved in test-glue/run-0/checkpoint-535/config.json
# Model weights saved in test-glue/run-0/checkpoint-535/pytorch_model.bin
# tokenizer config file saved in test-glue/run-0/checkpoint-535/tokenizer_config.json
# Special tokens file saved in test-glue/run-0/checkpoint-535/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.3534, 'learning_rate': 6.903384967527801e-06, 'epoch': 1.87}
#  67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                       | 1070/1605 [06:52<03:54,  2.28it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5046094059944153, 'eval_matthews_correlation': 0.5151128338364405, 'eval_runtime': 3.4797, 'eval_samples_per_second': 299.74, 'eval_steps_per_second': 4.885, 'epoch': 2.0}
#  67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                       | 1070/1605 [06:55<03:54,  2.28it/s]Saving model checkpoint to test-glue/run-0/checkpoint-1070
# Configuration saved in test-glue/run-0/checkpoint-1070/config.json
# Model weights saved in test-glue/run-0/checkpoint-1070/pytorch_model.bin
# tokenizer config file saved in test-glue/run-0/checkpoint-1070/tokenizer_config.json
# Special tokens file saved in test-glue/run-0/checkpoint-1070/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.2321, 'learning_rate': 1.1981081348601968e-06, 'epoch': 2.8}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1605/1605 [10:50<00:00,  2.47it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.6115336418151855, 'eval_matthews_correlation': 0.5150075229081177, 'eval_runtime': 3.4531, 'eval_samples_per_second': 302.048, 'eval_steps_per_second': 4.923, 'epoch': 3.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1605/1605 [10:53<00:00,  2.47it/s]Saving model checkpoint to test-glue/run-0/checkpoint-1605
# Configuration saved in test-glue/run-0/checkpoint-1605/config.json
# Model weights saved in test-glue/run-0/checkpoint-1605/pytorch_model.bin
# tokenizer config file saved in test-glue/run-0/checkpoint-1605/tokenizer_config.json
# Special tokens file saved in test-glue/run-0/checkpoint-1605/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/run-0/checkpoint-1070 (score: 0.5151128338364405).
# {'train_runtime': 663.4005, 'train_samples_per_second': 38.669, 'train_steps_per_second': 2.419, 'train_loss': 0.36322256783458673, 'epoch': 3.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1605/1605 [11:03<00:00,  2.42it/s]
# [I 2022-05-06 17:48:41,177] Trial 0 finished with value: 0.5150075229081177 and parameters: {'learning_rate': 1.831393863286301e-05, 'num_train_epochs': 3, 'seed': 29, 'per_device_train_batch_size': 4}. Best is trial 0 with value: 0.5150075229081177.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 5
#   Instantaneous batch size per device = 8
#   Total train batch size (w. parallel, distributed & accumulation) = 32
#   Gradient Accumulation steps = 1
#   Total optimization steps = 1340
#   0%|                                                                                                                                                                                 | 0/1340 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  20%|█████████████████████████████████▍                                                                                                                                     | 268/1340 [01:56<08:06,  2.21it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.4575524628162384, 'eval_matthews_correlation': 0.48293884127044134, 'eval_runtime': 3.2562, 'eval_samples_per_second': 320.312, 'eval_steps_per_second': 5.221, 'epoch': 1.0}
#  20%|█████████████████████████████████▍                                                                                                                                     | 268/1340 [01:59<08:06,  2.21it/s]Saving model checkpoint to test-glue/run-1/checkpoint-268
# Configuration saved in test-glue/run-1/checkpoint-268/config.json
# Model weights saved in test-glue/run-1/checkpoint-268/pytorch_model.bin
# tokenizer config file saved in test-glue/run-1/checkpoint-268/tokenizer_config.json
# Special tokens file saved in test-glue/run-1/checkpoint-268/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.4226, 'learning_rate': 2.575443678221771e-05, 'epoch': 1.87}
#  40%|██████████████████████████████████████████████████████████████████▊                                                                                                    | 536/1340 [04:05<05:19,  2.52it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.46979832649230957, 'eval_matthews_correlation': 0.520875943143754, 'eval_runtime': 3.3241, 'eval_samples_per_second': 313.768, 'eval_steps_per_second': 5.114, 'epoch': 2.0}
#  40%|██████████████████████████████████████████████████████████████████▊                                                                                                    | 536/1340 [04:08<05:19,  2.52it/s]Saving model checkpoint to test-glue/run-1/checkpoint-536
# Configuration saved in test-glue/run-1/checkpoint-536/config.json
# Model weights saved in test-glue/run-1/checkpoint-536/pytorch_model.bin
# tokenizer config file saved in test-glue/run-1/checkpoint-536/tokenizer_config.json
# Special tokens file saved in test-glue/run-1/checkpoint-536/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                  | 804/1340 [06:14<03:39,  2.44it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.6615782976150513, 'eval_matthews_correlation': 0.5072806068941901, 'eval_runtime': 3.4729, 'eval_samples_per_second': 300.322, 'eval_steps_per_second': 4.895, 'epoch': 3.0}
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                  | 804/1340 [06:18<03:39,  2.44it/s]Saving model checkpoint to test-glue/run-1/checkpoint-804
# Configuration saved in test-glue/run-1/checkpoint-804/config.json
# Model weights saved in test-glue/run-1/checkpoint-804/pytorch_model.bin
# tokenizer config file saved in test-glue/run-1/checkpoint-804/tokenizer_config.json
# Special tokens file saved in test-glue/run-1/checkpoint-804/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.1618, 'learning_rate': 1.0424414888040502e-05, 'epoch': 3.73}
#  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 1072/1340 [08:23<01:57,  2.28it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.7565516233444214, 'eval_matthews_correlation': 0.5373457262127674, 'eval_runtime': 3.5208, 'eval_samples_per_second': 296.238, 'eval_steps_per_second': 4.828, 'epoch': 4.0}
#  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 1072/1340 [08:27<01:57,  2.28it/s]Saving model checkpoint to test-glue/run-1/checkpoint-1072
# Configuration saved in test-glue/run-1/checkpoint-1072/config.json
# Model weights saved in test-glue/run-1/checkpoint-1072/pytorch_model.bin
# tokenizer config file saved in test-glue/run-1/checkpoint-1072/tokenizer_config.json
# Special tokens file saved in test-glue/run-1/checkpoint-1072/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1340/1340 [10:30<00:00,  2.40it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.8489832282066345, 'eval_matthews_correlation': 0.5424244271777358, 'eval_runtime': 3.4976, 'eval_samples_per_second': 298.207, 'eval_steps_per_second': 4.861, 'epoch': 5.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1340/1340 [10:33<00:00,  2.40it/s]Saving model checkpoint to test-glue/run-1/checkpoint-1340
# Configuration saved in test-glue/run-1/checkpoint-1340/config.json
# Model weights saved in test-glue/run-1/checkpoint-1340/pytorch_model.bin
# tokenizer config file saved in test-glue/run-1/checkpoint-1340/tokenizer_config.json
# Special tokens file saved in test-glue/run-1/checkpoint-1340/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/run-1/checkpoint-1340 (score: 0.5424244271777358).
# {'train_runtime': 643.4715, 'train_samples_per_second': 66.444, 'train_steps_per_second': 2.082, 'train_loss': 0.23800194014364215, 'epoch': 5.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1340/1340 [10:43<00:00,  2.08it/s]
# [I 2022-05-06 17:59:25,639] Trial 1 finished with value: 0.5424244271777358 and parameters: {'learning_rate': 4.1084458676394915e-05, 'num_train_epochs': 5, 'seed': 25, 'per_device_train_batch_size': 8}. Best is trial 1 with value: 0.5424244271777358.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 4
#   Instantaneous batch size per device = 4
#   Total train batch size (w. parallel, distributed & accumulation) = 16
#   Gradient Accumulation steps = 1
#   Total optimization steps = 2140
#   0%|                                                                                                                                                                                 | 0/2140 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.5303, 'learning_rate': 4.267808785535054e-05, 'epoch': 0.93}
#  25%|█████████████████████████████████████████▊                                                                                                                             | 535/2140 [03:41<11:28,  2.33it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5132118463516235, 'eval_matthews_correlation': 0.3909403421099583, 'eval_runtime': 3.554, 'eval_samples_per_second': 293.469, 'eval_steps_per_second': 4.783, 'epoch': 1.0}
#  25%|█████████████████████████████████████████▊                                                                                                                             | 535/2140 [03:45<11:28,  2.33it/s]Saving model checkpoint to test-glue/run-2/checkpoint-535
# Configuration saved in test-glue/run-2/checkpoint-535/config.json
# Model weights saved in test-glue/run-2/checkpoint-535/pytorch_model.bin
# tokenizer config file saved in test-glue/run-2/checkpoint-535/tokenizer_config.json
# Special tokens file saved in test-glue/run-2/checkpoint-535/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.322, 'learning_rate': 2.966647570432903e-05, 'epoch': 1.87}
#  50%|███████████████████████████████████████████████████████████████████████████████████                                                                                   | 1070/2140 [07:39<07:26,  2.40it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5290344953536987, 'eval_matthews_correlation': 0.5024061937657525, 'eval_runtime': 3.386, 'eval_samples_per_second': 308.033, 'eval_steps_per_second': 5.021, 'epoch': 2.0}
#  50%|███████████████████████████████████████████████████████████████████████████████████                                                                                   | 1070/2140 [07:43<07:26,  2.40it/s]Saving model checkpoint to test-glue/run-2/checkpoint-1070
# Configuration saved in test-glue/run-2/checkpoint-1070/config.json
# Model weights saved in test-glue/run-2/checkpoint-1070/pytorch_model.bin
# tokenizer config file saved in test-glue/run-2/checkpoint-1070/tokenizer_config.json
# Special tokens file saved in test-glue/run-2/checkpoint-1070/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.1952, 'learning_rate': 1.6654863553307526e-05, 'epoch': 2.8}
#  75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                         | 1605/2140 [11:36<03:41,  2.42it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.8275901675224304, 'eval_matthews_correlation': 0.4869134372691264, 'eval_runtime': 3.3655, 'eval_samples_per_second': 309.907, 'eval_steps_per_second': 5.051, 'epoch': 3.0}
#  75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                         | 1605/2140 [11:39<03:41,  2.42it/s]Saving model checkpoint to test-glue/run-2/checkpoint-1605
# Configuration saved in test-glue/run-2/checkpoint-1605/config.json
# Model weights saved in test-glue/run-2/checkpoint-1605/pytorch_model.bin
# tokenizer config file saved in test-glue/run-2/checkpoint-1605/tokenizer_config.json
# Special tokens file saved in test-glue/run-2/checkpoint-1605/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.1168, 'learning_rate': 3.643251402286021e-06, 'epoch': 3.74}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2140/2140 [15:31<00:00,  2.45it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.9902224540710449, 'eval_matthews_correlation': 0.4986084626480731, 'eval_runtime': 3.3589, 'eval_samples_per_second': 310.516, 'eval_steps_per_second': 5.061, 'epoch': 4.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2140/2140 [15:35<00:00,  2.45it/s]Saving model checkpoint to test-glue/run-2/checkpoint-2140
# Configuration saved in test-glue/run-2/checkpoint-2140/config.json
# Model weights saved in test-glue/run-2/checkpoint-2140/pytorch_model.bin
# tokenizer config file saved in test-glue/run-2/checkpoint-2140/tokenizer_config.json
# Special tokens file saved in test-glue/run-2/checkpoint-2140/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/run-2/checkpoint-1070 (score: 0.5024061937657525).
# {'train_runtime': 945.1939, 'train_samples_per_second': 36.187, 'train_steps_per_second': 2.264, 'train_loss': 0.2784071797522429, 'epoch': 4.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2140/2140 [15:45<00:00,  2.26it/s]
# [I 2022-05-06 18:15:11,825] Trial 2 finished with value: 0.4986084626480731 and parameters: {'learning_rate': 5.568970000637204e-05, 'num_train_epochs': 4, 'seed': 28, 'per_device_train_batch_size': 4}. Best is trial 1 with value: 0.5424244271777358.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 1
#   Instantaneous batch size per device = 32
#   Total train batch size (w. parallel, distributed & accumulation) = 128
#   Gradient Accumulation steps = 1
#   Total optimization steps = 67
#   0%|                                                                                                                                                                                   | 0/67 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:42<00:00,  1.55it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.6770498752593994, 'eval_matthews_correlation': 0.01348864658799917, 'eval_runtime': 3.5062, 'eval_samples_per_second': 297.47, 'eval_steps_per_second': 4.848, 'epoch': 1.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:46<00:00,  1.55it/s]Saving model checkpoint to test-glue/run-3/checkpoint-67
# Configuration saved in test-glue/run-3/checkpoint-67/config.json
# Model weights saved in test-glue/run-3/checkpoint-67/pytorch_model.bin
# tokenizer config file saved in test-glue/run-3/checkpoint-67/tokenizer_config.json
# Special tokens file saved in test-glue/run-3/checkpoint-67/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/run-3/checkpoint-67 (score: 0.01348864658799917).
# {'train_runtime': 56.3181, 'train_samples_per_second': 151.834, 'train_steps_per_second': 1.19, 'train_loss': 0.6917540137447528, 'epoch': 1.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:56<00:00,  1.19it/s]
# [I 2022-05-06 18:16:09,098] Trial 3 finished with value: 0.01348864658799917 and parameters: {'learning_rate': 1.0772672038057411e-06, 'num_train_epochs': 1, 'seed': 36, 'per_device_train_batch_size': 32}. Best is trial 1 with value: 0.5424244271777358.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 5
#   Instantaneous batch size per device = 16
#   Total train batch size (w. parallel, distributed & accumulation) = 64
#   Gradient Accumulation steps = 1
#   Total optimization steps = 670
#   0%|                                                                                                                                                                                  | 0/670 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  20%|█████████████████████████████████▌                                                                                                                                      | 134/670 [01:16<04:51,  1.84it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.6079733371734619, 'eval_matthews_correlation': 0.0, 'eval_runtime': 3.3488, 'eval_samples_per_second': 311.451, 'eval_steps_per_second': 5.076, 'epoch': 1.0}
#  20%|█████████████████████████████████▌                                                                                                                                      | 134/670 [01:19<04:51,  1.84it/s]Saving model checkpoint to test-glue/run-4/checkpoint-134
# Configuration saved in test-glue/run-4/checkpoint-134/config.json
# Model weights saved in test-glue/run-4/checkpoint-134/pytorch_model.bin
# tokenizer config file saved in test-glue/run-4/checkpoint-134/tokenizer_config.json
# Special tokens file saved in test-glue/run-4/checkpoint-134/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  40%|███████████████████████████████████████████████████████████████████▏                                                                                                    | 268/670 [02:46<03:32,  1.89it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5886809825897217, 'eval_matthews_correlation': 0.0, 'eval_runtime': 3.1923, 'eval_samples_per_second': 326.722, 'eval_steps_per_second': 5.325, 'epoch': 2.0}
#  40%|███████████████████████████████████████████████████████████████████▏                                                                                                    | 268/670 [02:49<03:32,  1.89it/s]Saving model checkpoint to test-glue/run-4/checkpoint-268
# Configuration saved in test-glue/run-4/checkpoint-268/config.json
# Model weights saved in test-glue/run-4/checkpoint-268/pytorch_model.bin
# tokenizer config file saved in test-glue/run-4/checkpoint-268/tokenizer_config.json
# Special tokens file saved in test-glue/run-4/checkpoint-268/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                   | 402/670 [04:15<02:22,  1.88it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.570699155330658, 'eval_matthews_correlation': 0.0463559874942472, 'eval_runtime': 3.278, 'eval_samples_per_second': 318.18, 'eval_steps_per_second': 5.186, 'epoch': 3.0}
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                   | 402/670 [04:18<02:22,  1.88it/s]Saving model checkpoint to test-glue/run-4/checkpoint-402
# Configuration saved in test-glue/run-4/checkpoint-402/config.json
# Model weights saved in test-glue/run-4/checkpoint-402/pytorch_model.bin
# tokenizer config file saved in test-glue/run-4/checkpoint-402/tokenizer_config.json
# Special tokens file saved in test-glue/run-4/checkpoint-402/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.5951, 'learning_rate': 3.9680065114238564e-07, 'epoch': 3.73}
#  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                 | 536/670 [05:45<01:19,  1.69it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5631893873214722, 'eval_matthews_correlation': 0.0463559874942472, 'eval_runtime': 3.369, 'eval_samples_per_second': 309.588, 'eval_steps_per_second': 5.046, 'epoch': 4.0}
#  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                 | 536/670 [05:48<01:19,  1.69it/s]Saving model checkpoint to test-glue/run-4/checkpoint-536
# Configuration saved in test-glue/run-4/checkpoint-536/config.json
# Model weights saved in test-glue/run-4/checkpoint-536/pytorch_model.bin
# tokenizer config file saved in test-glue/run-4/checkpoint-536/tokenizer_config.json
# Special tokens file saved in test-glue/run-4/checkpoint-536/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 670/670 [07:14<00:00,  1.88it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.560096800327301, 'eval_matthews_correlation': 0.0592680243795702, 'eval_runtime': 3.5552, 'eval_samples_per_second': 293.369, 'eval_steps_per_second': 4.782, 'epoch': 5.0}
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 670/670 [07:18<00:00,  1.88it/s]Saving model checkpoint to test-glue/run-4/checkpoint-670
# Configuration saved in test-glue/run-4/checkpoint-670/config.json
# Model weights saved in test-glue/run-4/checkpoint-670/pytorch_model.bin
# tokenizer config file saved in test-glue/run-4/checkpoint-670/tokenizer_config.json
# Special tokens file saved in test-glue/run-4/checkpoint-670/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/run-4/checkpoint-670 (score: 0.0592680243795702).
# {'train_runtime': 448.0325, 'train_samples_per_second': 95.428, 'train_steps_per_second': 1.495, 'train_loss': 0.5835773638824918, 'epoch': 5.0}
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 670/670 [07:28<00:00,  1.50it/s]
# [I 2022-05-06 18:23:38,208] Trial 4 finished with value: 0.0592680243795702 and parameters: {'learning_rate': 1.563861389796461e-06, 'num_train_epochs': 5, 'seed': 26, 'per_device_train_batch_size': 16}. Best is trial 1 with value: 0.5424244271777358.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 2
#   Instantaneous batch size per device = 16
#   Total train batch size (w. parallel, distributed & accumulation) = 64
#   Gradient Accumulation steps = 1
#   Total optimization steps = 268
#   0%|                                                                                                                                                                                  | 0/268 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  50%|████████████████████████████████████████████████████████████████████████████████████                                                                                    | 134/268 [01:16<01:11,  1.89it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.6102632880210876, 'eval_matthews_correlation': 0.0, 'eval_runtime': 3.4864, 'eval_samples_per_second': 299.164, 'eval_steps_per_second': 4.876, 'epoch': 1.0}
#  50%|████████████████████████████████████████████████████████████████████████████████████                                                                                    | 134/268 [01:19<01:19,  1.68it/s]
# [I 2022-05-06 18:24:58,758] Trial 5 pruned.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 4
#   Instantaneous batch size per device = 32
#   Total train batch size (w. parallel, distributed & accumulation) = 128
#   Gradient Accumulation steps = 1
#   Total optimization steps = 268
#   0%|                                                                                                                                                                                  | 0/268 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  25%|██████████████████████████████████████████▎                                                                                                                              | 67/268 [00:43<02:16,  1.48it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.510810375213623, 'eval_matthews_correlation': 0.42696977338948733, 'eval_runtime': 3.3535, 'eval_samples_per_second': 311.019, 'eval_steps_per_second': 5.069, 'epoch': 1.0}
#  25%|██████████████████████████████████████████▎                                                                                                                              | 67/268 [00:46<02:16,  1.48it/s]Saving model checkpoint to test-glue/run-6/checkpoint-67
# Configuration saved in test-glue/run-6/checkpoint-67/config.json
# Model weights saved in test-glue/run-6/checkpoint-67/pytorch_model.bin
# tokenizer config file saved in test-glue/run-6/checkpoint-67/tokenizer_config.json
# Special tokens file saved in test-glue/run-6/checkpoint-67/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  50%|████████████████████████████████████████████████████████████████████████████████████                                                                                    | 134/268 [01:39<01:23,  1.60it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5006619691848755, 'eval_matthews_correlation': 0.4948269680739418, 'eval_runtime': 3.4964, 'eval_samples_per_second': 298.306, 'eval_steps_per_second': 4.862, 'epoch': 2.0}
#  50%|████████████████████████████████████████████████████████████████████████████████████                                                                                    | 134/268 [01:43<01:43,  1.30it/s]
# [I 2022-05-06 18:26:43,327] Trial 6 pruned.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 4
#   Instantaneous batch size per device = 8
#   Total train batch size (w. parallel, distributed & accumulation) = 32
#   Gradient Accumulation steps = 1
#   Total optimization steps = 1072
#   0%|                                                                                                                                                                                 | 0/1072 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  25%|█████████████████████████████████████████▊                                                                                                                             | 268/1072 [01:53<06:00,  2.23it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.4914335310459137, 'eval_matthews_correlation': 0.44075455043157863, 'eval_runtime': 3.3338, 'eval_samples_per_second': 312.86, 'eval_steps_per_second': 5.099, 'epoch': 1.0}
#  25%|█████████████████████████████████████████▊                                                                                                                             | 268/1072 [01:57<06:00,  2.23it/s]Saving model checkpoint to test-glue/run-7/checkpoint-268
# Configuration saved in test-glue/run-7/checkpoint-268/config.json
# Model weights saved in test-glue/run-7/checkpoint-268/pytorch_model.bin
# tokenizer config file saved in test-glue/run-7/checkpoint-268/tokenizer_config.json
# Special tokens file saved in test-glue/run-7/checkpoint-268/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.4183, 'learning_rate': 4.05880046786775e-05, 'epoch': 1.87}
#  50%|███████████████████████████████████████████████████████████████████████████████████▌                                                                                   | 536/1072 [04:01<03:48,  2.34it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5417600274085999, 'eval_matthews_correlation': 0.4389049261444844, 'eval_runtime': 3.4956, 'eval_samples_per_second': 298.375, 'eval_steps_per_second': 4.863, 'epoch': 2.0}
#  50%|███████████████████████████████████████████████████████████████████████████████████▌                                                                                   | 536/1072 [04:04<04:04,  2.19it/s]
# [I 2022-05-06 18:30:49,212] Trial 7 pruned.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 2
#   Instantaneous batch size per device = 4
#   Total train batch size (w. parallel, distributed & accumulation) = 16
#   Gradient Accumulation steps = 1
#   Total optimization steps = 1070
#   0%|                                                                                                                                                                                 | 0/1070 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.5981, 'learning_rate': 1.2571251970139977e-06, 'epoch': 0.93}
#  50%|███████████████████████████████████████████████████████████████████████████████████▌                                                                                   | 535/1070 [03:42<03:40,  2.43it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5605311393737793, 'eval_matthews_correlation': 0.0463559874942472, 'eval_runtime': 3.3598, 'eval_samples_per_second': 310.435, 'eval_steps_per_second': 5.06, 'epoch': 1.0}
#  50%|███████████████████████████████████████████████████████████████████████████████████▌                                                                                   | 535/1070 [03:45<03:45,  2.37it/s]
# [I 2022-05-06 18:34:35,982] Trial 8 pruned.
# Trial:
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 4
#   Instantaneous batch size per device = 4
#   Total train batch size (w. parallel, distributed & accumulation) = 16
#   Gradient Accumulation steps = 1
#   Total optimization steps = 2140
#   0%|                                                                                                                                                                                 | 0/2140 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.5663, 'learning_rate': 7.241449414448643e-05, 'epoch': 0.93}
#  25%|█████████████████████████████████████████▊                                                                                                                             | 535/2140 [03:43<10:22,  2.58it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.5107989311218262, 'eval_matthews_correlation': 0.3603922210193049, 'eval_runtime': 3.4502, 'eval_samples_per_second': 302.297, 'eval_steps_per_second': 4.927, 'epoch': 1.0}
#  25%|█████████████████████████████████████████▊                                                                                                                             | 535/2140 [03:47<11:21,  2.36it/s]
# [I 2022-05-06 18:38:24,154] Trial 9 pruned.
# BestRun(run_id='1', objective=0.5424244271777358, hyperparameters={'learning_rate': 4.1084458676394915e-05, 'num_train_epochs': 5, 'seed': 25, 'per_device_train_batch_size': 8})
# loading configuration file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/config.json
# Model config DistilBertConfig {
#   "_name_or_path": "/data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased",
#   "activation": "gelu",
#   "architectures": [
#     "DistilBertForMaskedLM"
#   ],
#   "attention_dropout": 0.1,
#   "dim": 768,
#   "dropout": 0.1,
#   "hidden_dim": 3072,
#   "initializer_range": 0.02,
#   "max_position_embeddings": 512,
#   "model_type": "distilbert",
#   "n_heads": 12,
#   "n_layers": 6,
#   "pad_token_id": 0,
#   "qa_dropout": 0.1,
#   "seq_classif_dropout": 0.2,
#   "sinusoidal_pos_embds": false,
#   "tie_weights_": true,
#   "transformers_version": "4.18.0",
#   "vocab_size": 30522
# }
#
# loading weights file /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased/pytorch_model.bin
# Some weights of the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_projector.bias']
# - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at /data0/nfs_data/zhaoxi9/pretrained_language_model/huggingface_pretrained_models/distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# The following columns in the training set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   FutureWarning,
# ***** Running training *****
#   Num examples = 8551
#   Num Epochs = 5
#   Instantaneous batch size per device = 8
#   Total train batch size (w. parallel, distributed & accumulation) = 32
#   Gradient Accumulation steps = 1
#   Total optimization steps = 1340
#   0%|                                                                                                                                                                                 | 0/1340 [00:00<?, ?it/s]/data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  20%|█████████████████████████████████▍                                                                                                                                     | 268/1340 [01:54<07:58,  2.24it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.4575524628162384, 'eval_matthews_correlation': 0.48293884127044134, 'eval_runtime': 3.3144, 'eval_samples_per_second': 314.687, 'eval_steps_per_second': 5.129, 'epoch': 1.0}
#  20%|█████████████████████████████████▍                                                                                                                                     | 268/1340 [01:57<07:58,  2.24it/s]Saving model checkpoint to test-glue/checkpoint-268
# Configuration saved in test-glue/checkpoint-268/config.json
# Model weights saved in test-glue/checkpoint-268/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-268/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-268/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.4226, 'learning_rate': 2.575443678221771e-05, 'epoch': 1.87}
#  40%|██████████████████████████████████████████████████████████████████▊                                                                                                    | 536/1340 [04:01<05:33,  2.41it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.46979832649230957, 'eval_matthews_correlation': 0.520875943143754, 'eval_runtime': 3.3989, 'eval_samples_per_second': 306.863, 'eval_steps_per_second': 5.002, 'epoch': 2.0}
#  40%|██████████████████████████████████████████████████████████████████▊                                                                                                    | 536/1340 [04:04<05:33,  2.41it/s]Saving model checkpoint to test-glue/checkpoint-536
# Configuration saved in test-glue/checkpoint-536/config.json
# Model weights saved in test-glue/checkpoint-536/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-536/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-536/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                  | 804/1340 [06:08<03:43,  2.40it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.6615782976150513, 'eval_matthews_correlation': 0.5072806068941901, 'eval_runtime': 3.2007, 'eval_samples_per_second': 325.869, 'eval_steps_per_second': 5.311, 'epoch': 3.0}
#  60%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                  | 804/1340 [06:11<03:43,  2.40it/s]Saving model checkpoint to test-glue/checkpoint-804
# Configuration saved in test-glue/checkpoint-804/config.json
# Model weights saved in test-glue/checkpoint-804/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-804/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-804/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# {'loss': 0.1618, 'learning_rate': 1.0424414888040502e-05, 'epoch': 3.73}
#  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 1072/1340 [08:13<01:50,  2.42it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.7565516233444214, 'eval_matthews_correlation': 0.5373457262127674, 'eval_runtime': 3.5244, 'eval_samples_per_second': 295.933, 'eval_steps_per_second': 4.823, 'epoch': 4.0}
#  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 1072/1340 [08:17<01:50,  2.42it/s]Saving model checkpoint to test-glue/checkpoint-1072
# Configuration saved in test-glue/checkpoint-1072/config.json
# Model weights saved in test-glue/checkpoint-1072/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-1072/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-1072/special_tokens_map.json
# /data0/zhaoxi9/miniconda3/envs/py37tch111/lib/python3.7/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
#   warnings.warn('Was asked to gather along dimension 0, but all '
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1340/1340 [10:21<00:00,  2.56it/s]The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence. If idx, sentence are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
# ***** Running Evaluation *****
#   Num examples = 1043
#   Batch size = 16
# {'eval_loss': 0.8489832282066345, 'eval_matthews_correlation': 0.5424244271777358, 'eval_runtime': 3.5322, 'eval_samples_per_second': 295.287, 'eval_steps_per_second': 4.813, 'epoch': 5.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1340/1340 [10:24<00:00,  2.56it/s]Saving model checkpoint to test-glue/checkpoint-1340
# Configuration saved in test-glue/checkpoint-1340/config.json
# Model weights saved in test-glue/checkpoint-1340/pytorch_model.bin
# tokenizer config file saved in test-glue/checkpoint-1340/tokenizer_config.json
# Special tokens file saved in test-glue/checkpoint-1340/special_tokens_map.json
#
#
# Training completed. Do not forget to share your model on huggingface.co/models =)
#
#
# Loading best model from test-glue/checkpoint-1340 (score: 0.5424244271777358).
# {'train_runtime': 637.3623, 'train_samples_per_second': 67.081, 'train_steps_per_second': 2.102, 'train_loss': 0.23800194014364215, 'epoch': 5.0}
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1340/1340 [10:37<00:00,  2.10it/s]