#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import math
import torch
from torch import nn
import ipdb

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pickle
from datasets import load_dataset
from copy import deepcopy

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )

from transformers.trainer_utils import get_last_checkpoint

from cl_collator import DataCollator
from cl_dataset import gen_cache_path

from assets import task_config, lora_state_dict_A, lora_state_dict_B

from cl_trainer_gainlora_inflora import DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
from datasets.download import DownloadConfig

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

local_data_path = "/home/work/nltk_data"
nltk.data.path.append(local_data_path)

@dataclass(frozen=False)
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )

    prefix_len: Optional[int] = field(
        default=10,
        metadata={
            "help": "Length of Prompt."
        },
    )

    mlp_hidden_dim: Optional[int] = field(
        default=100,
        metadata={
            "help": "Intrinsic dimension of the latent MLP space."
        },
    )

    attn_temperature: Optional[int] = field(
        default=1,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_alpha: Optional[int] = field(
        default=1,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )

    run_single: bool = field(
        default=False,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )

    previous_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "the path to load previous prompts."}
    )

    previous_prompt_key_path: Optional[str] = field(
        default=None,
        metadata={"help": "the path to load previous prompts."}
    )

    load_checkpoint_from: str = field(
        default=None,
        metadata={"help": "Path to load previous checkpoints"}
    )


@dataclass(frozen=False)
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the CL train/dev/test splits."}
    )
    gen_data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the generated train/dev/test splits."}
    )
    task_order: str = field(
        default=None, metadata={"help": "order of the tasks"}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    replay_task_list: Optional[str] = field(
        default='', metadata={
            "help": "Different tasks to replay"
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    add_instruction_replay: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend definition and few-shot cases before the task input during replay."}
    )

@dataclass(frozen=False)
class TrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    lamda_1: float = field(default = 0.5)
    lamda_2: float = field(default = 0)
    
    kl_ratio: Optional[float] = field(
        default=0.5,
        metadata={"help": "ratio of the replay kl loss"}
    )
    data_replay_freq: Optional[int] = field(
        default=-1,
        metadata={"help": "replay frequency"}
    )
    replay_after_n_epoch: Optional[int] = field(
        default=0,
        metadata={"help": "replay after n epoch"}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
    )
    attn_lr: Optional[float] = field(
        default=0,
        metadata={"help": "learning rate of the attention module"}
    )
    model_name: Optional[str] = field(
        default='SAPT',
        metadata={"help": "models' name"}
    )
    chunk: Optional[int] = field(
        default=1,
        metadata={"help": "models' name"}
    )
    threshold: Optional[float] = field(
        default=0.99,
        metadata={"help": "learning rate of the attention module"}
    )
    transthreshold: Optional[float] = field(
        default=0.99,
        metadata={"help": "learning rate of the attention module"}
    )
    op: Optional[int] = field(
        default=0,
        metadata={"help": "occupy"}
    )
    persent: Optional[float] = field(
        default=1.0,
        metadata={"help": "persent"}
    )
    n_groups: Optional[int] = field(
        default=1,
        metadata={"help": "persent"}
    )
    lambda1: Optional[float] = field(
        default=1.0,
        metadata={"help": "persent"}
    )
    lambda2: Optional[float] = field(
        default=1.0,
        metadata={"help": "persent"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args._frozen = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    task_order = data_args.task_order.split(',')
    
    cur_task = data_args.task_config_dir.split('/')[-1]
    cur_task_id = task_order.index(cur_task)

    download_config = DownloadConfig
    download_config.local_files_only = True
    # Get the CL dataset
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "cl_dataset.py"),
        data_dir=data_args.data_dir,
        download_config=download_config,
        task_config_dir=data_args.task_config_dir,
        # cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )
    raw_datasets.cleanup_cache_files()

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    prompt_config = {
        'seq_len': data_args.max_source_length,
        'mlp_hidden_dim': model_args.mlp_hidden_dim,
        'attn_temperature': model_args.attn_temperature,
        'previous_lora_path': model_args.previous_lora_path,
        'previous_prompt_key_path': model_args.previous_prompt_key_path,
        'task_id': cur_task_id,
        'run_single': model_args.run_single,
        'lora_r': model_args.lora_r,
        'lora_alpha': model_args.lora_alpha,
        'lora_dropout': model_args.lora_dropout
    }

    if training_args.model_name in ['gainlora_inflora']:
        from t5_large import T5ForConditionalGeneration
    elif training_args.model_name in ['inflora']:
        from t5_inflora import T5ForConditionalGeneration
    else:
        raise NotImplementedError

    model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            prompt_config,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model.persent = training_args.persent
    model.resize_token_embeddings(len(tokenizer))

    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
    except:
        device = torch.device(f"cuda:0")
    if model_args.load_checkpoint_from and training_args.model_name in ['gainlora_inflora']:
        print("----------Loading Previous Query Projection Layer----------")
        model.encoder.trans_input.load_state_dict(torch.load(model_args.load_checkpoint_from, map_location=device))
        # ipdb.set_trace()
        model.encoder.previous_trans_input.input_linear[0].data.copy_(torch.load(model_args.load_checkpoint_from, map_location=device)['0.weight'])
        model.encoder.previous_trans_input.output_linear[0].data.copy_(torch.load(model_args.load_checkpoint_from, map_location=device)['2.weight'])
        model.encoder.previous_trans_input.state_dict()
        if cur_task_id > 1:
            model.encoder.previous_trans_input.input_linear[1:].data.copy_(torch.load(model_args.load_checkpoint_from.replace('trans_input.pt', 'previous_trans_input.pt'), map_location=device)['input_linear'])
            model.encoder.previous_trans_input.output_linear[1:].data.copy_(torch.load(model_args.load_checkpoint_from.replace('trans_input.pt', 'previous_trans_input.pt'), map_location=device)['output_linear'])
        print("----------Loading Previous Query Projection Layer Done----------")

    if model_args.previous_lora_path:
        previous_lora_list = model_args.previous_lora_path.split(',')
        previous_lora_list.reverse()
        print(previous_lora_list)
        print("----------Loading Previous LoRA Weights----------")
        for i, path in enumerate(previous_lora_list):
            lora_A = torch.load(os.path.join(path, "lora_weights_A.pt"), map_location=device)
            lora_B = torch.load(os.path.join(path, "lora_weights_B.pt"), map_location=device)
            ## Encoder Layer       
            for j in range(config.num_layers):
                model.encoder.block[j].layer[0].SelfAttention.previous_lora_weights_q[i].lora_A.data.copy_(
                    lora_A[f"encoder.block.{j}.layer.0.SelfAttention.lora_q.lora_A"]
                )
                model.encoder.block[j].layer[0].SelfAttention.previous_lora_weights_q[i].lora_B.data.copy_(
                    lora_B[f"encoder.block.{j}.layer.0.SelfAttention.lora_q.lora_B"]
                )
                model.encoder.block[j].layer[0].SelfAttention.previous_lora_weights_v[i].lora_A.data.copy_(
                    lora_A[f"encoder.block.{j}.layer.0.SelfAttention.lora_v.lora_A"]
                )
                model.encoder.block[j].layer[0].SelfAttention.previous_lora_weights_v[i].lora_B.data.copy_(
                    lora_B[f"encoder.block.{j}.layer.0.SelfAttention.lora_v.lora_B"]
                )
            ## Decoder Layaer
            for j in range(config.num_layers):
                model.decoder.block[j].layer[0].SelfAttention.previous_lora_weights_q[i].lora_A.data.copy_(
                    lora_A[f"decoder.block.{j}.layer.0.SelfAttention.lora_q.lora_A"]
                )
                model.decoder.block[j].layer[0].SelfAttention.previous_lora_weights_q[i].lora_B.data.copy_(
                    lora_B[f"decoder.block.{j}.layer.0.SelfAttention.lora_q.lora_B"]
                )
                model.decoder.block[j].layer[0].SelfAttention.previous_lora_weights_v[i].lora_A.data.copy_(
                    lora_A[f"decoder.block.{j}.layer.0.SelfAttention.lora_v.lora_A"]
                )
                model.decoder.block[j].layer[0].SelfAttention.previous_lora_weights_v[i].lora_B.data.copy_(
                    lora_B[f"decoder.block.{j}.layer.0.SelfAttention.lora_v.lora_B"]
                )
                model.decoder.block[j].layer[1].EncDecAttention.previous_lora_weights_q[i].lora_A.data.copy_(
                    lora_A[f"decoder.block.{j}.layer.1.EncDecAttention.lora_q.lora_A"]
                )
                model.decoder.block[j].layer[1].EncDecAttention.previous_lora_weights_q[i].lora_B.data.copy_(
                    lora_B[f"decoder.block.{j}.layer.1.EncDecAttention.lora_q.lora_B"]
                )
                model.decoder.block[j].layer[1].EncDecAttention.previous_lora_weights_v[i].lora_A.data.copy_(
                    lora_A[f"decoder.block.{j}.layer.1.EncDecAttention.lora_v.lora_A"]
                )
                model.decoder.block[j].layer[1].EncDecAttention.previous_lora_weights_v[i].lora_B.data.copy_(
                    lora_B[f"decoder.block.{j}.layer.1.EncDecAttention.lora_v.lora_B"]
                )

    for name, param in model.named_parameters():
        if  training_args.model_name in ['gainlora_olora']:
            param.requires_grad = False
            if ("lora" in name and "previous_lora_weights" not in name) or ("trans_input" in name and "previous_trans_input" not in name) or "prompt_key" in name:
                param.requires_grad = True
        elif training_args.model_name in ['gainlora_inflora', 'inflora']:
            param.requires_grad = False
            if ("lora_B" in name and "previous_lora_weights" not in name) or ("trans_input" in name and "previous_trans_input" not in name) or "prompt_key" in name:
                param.requires_grad = True

    total_params, params = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
            total_params += p.numel()
        params += p.numel()

    print(
        "Total number of parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    replay_dataset_dict, replay_label_dict = None, None
    data_collator_replay = None
    if training_args.data_replay_freq != -1:
        data_dir = data_args.gen_data_dir
        data_collator_replay = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_dataset_name=data_args.add_dataset_name,
            add_instruction_replay=data_args.add_instruction_replay,
            num_examples=data_args.num_examples,
            input_record_file=data_args.input_record_file)

        replay_dataset_dict, replay_label_dict = None, None
        if model_args.load_checkpoint_from:
            replay_dataset_dict = {}
            for idx in range(cur_task_id):
                raw_datasets_gen = load_dataset(
                    os.path.join(CURRENT_DIR, "cl_dataset.py"),
                    data_dir=data_dir,
                    download_config=download_config,
                    task_config_dir=task_config[task_order[idx]],
                    cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
                    max_num_instances_per_task=data_args.max_num_instances_per_task,
                    max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
                    num_examples=data_args.num_examples)

                replay_dataset_dict[task_order[idx]] = raw_datasets_gen["train"]
                print(raw_datasets_gen)

            replay_label_dict = {}
            for idx in range(0,cur_task_id):
                with open(os.path.join("../logs_and_outputs/" + training_args.run_name + "/outputs/", str(idx+1)+"-"+task_order[idx], "saved_weights", "attention_weights.pkl"), 'rb') as f:
                    attn_weights = pickle.load(f)
                replay_label_dict[task_order[idx]] = torch.cat([torch.tensor([0.] * (cur_task_id - idx)), torch.tensor(attn_weights)], dim=0).to(dtype=torch.bfloat16, device='cuda')
            print(replay_label_dict)
        print('-'*50)

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result
    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    training_args.step_per_epoch = math.ceil(len(raw_datasets["train"]) / training_args.per_device_train_batch_size / world_size / training_args.gradient_accumulation_steps)
    training_args.eval_steps = 5 * training_args.step_per_epoch
    training_args.save_steps = 5 * training_args.step_per_epoch

    if training_args.model_name in ['gainlora_inflora', 'inflora']:
        for module in model.modules():
            if hasattr(module, 'get_feature'):
                module.get_chunk(training_args.chunk)
        if training_args.model_name in ['gainlora_inflora']:
            model.encoder.get_chunk(training_args.chunk)
    else:
        raise NotImplementedError
    if training_args.model_name == 'gainlora_inflora':
        from cl_trainer_gainlora_inflora import GainLoRA_InfLoRA_Trainer
        trainer = GainLoRA_InfLoRA_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            cur_task_id=cur_task_id,
            task_order=task_order,
            data_collator_replay=data_collator_replay,
            replay_dataset_dict=replay_dataset_dict,
            replay_label_dict=replay_label_dict,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_rouge_metrics,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
        )
        if training_args.do_train:
            trainer.get_reg_matrix()
    elif training_args.model_name == 'inflora':
        from cl_trainer_inflora import InfLoRA_Trainer
        trainer = InfLoRA_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            cur_task_id=cur_task_id,
            task_order=task_order,
            data_collator_replay=data_collator_replay,
            replay_dataset_dict=replay_dataset_dict,
            replay_label_dict=replay_label_dict,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_rouge_metrics,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
        )
        if training_args.do_train:
            trainer.get_reg_matrix()
    else:
        raise NotImplementedError

    trainer.is_deepspeed_enabled = False
    print("is_deepspeed_enabled", trainer.is_deepspeed_enabled)

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        save_path = training_args.output_dir + "/saved_weights"
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except:
                pass

        if not prompt_config["run_single"]:
            if prompt_config["previous_prompt_key_path"] is not None and training_args.model_name in ['gainlora_inflora']:
                previous_trans_input = deepcopy(trainer.model.encoder.previous_trans_input.state_dict())
                torch.save(previous_trans_input, os.path.join(save_path, 'previous_trans_input.pt'))
            torch.save(trainer.model.encoder.trans_input.state_dict(), os.path.join(save_path, 'trans_input.pt'))
            
        
        if prompt_config["previous_prompt_key_path"] is not None:
            torch.save(lora_state_dict_A(model, task_name=cur_task), os.path.join(save_path, 'lora_weights_A.pt'))
            torch.save(lora_state_dict_B(model, task_name=cur_task), os.path.join(save_path, 'lora_weights_B.pt'))
            if not prompt_config["run_single"]:
                torch.save(torch.cat([trainer.model.encoder.prompt_key, trainer.model.encoder.previous_prompts_keys], dim=0).data, os.path.join(save_path, 'prompts_keys_till_now.pt'))
        else:
            torch.save(lora_state_dict_A(model, task_name=cur_task), os.path.join(save_path, 'lora_weights_A.pt'))
            torch.save(lora_state_dict_B(model, task_name=cur_task), os.path.join(save_path, 'lora_weights_B.pt'))
            if not prompt_config["run_single"]:
                torch.save(trainer.model.encoder.prompt_key.data, os.path.join(save_path, 'prompts_keys_till_now.pt'))
        tokenizer.save_pretrained(save_path)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

        trainer.get_repsentation()

    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict or training_args.do_train:
        print("*** Prediction ***")
        logger.info("*** Prediction ***")
        logger.info("*** Loading CheckPoint ***")

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        trainer.model.encoder.is_inference = True
        _ = trainer.predict(
            eval_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        
        if not prompt_config["run_single"]:
            # ipdb.set_trace()
            save_path = training_args.output_dir + "/saved_weights"
            with open(os.path.join(save_path, "attention_weights.pkl"), 'wb') as f:
                print("*"*20, "Saving Attention Weights", "*"*20)
                print(np.array(np.concatenate(trainer.model.encoder.all_attn_weights)).mean(axis=0))
                pickle.dump(np.array(np.concatenate(trainer.model.encoder.all_attn_weights)).mean(axis=0), f)
            trainer.model.encoder.is_inference = False

        if training_args.do_predict:


            if training_args.model_name in ["gainlora_inflora", "inflora"]:
                trainer.model.encoder.is_inference = True
                all_group_accs = []
                for task_id in range(cur_task_id + 1):
                    del trainer.model.encoder.all_attn_weights
                    trainer.model.encoder.all_attn_weights = []
                    select_indexs = [id for id, instance in enumerate(predict_dataset) if task_order.index(instance['Dataset']) == task_id]
                    select_predict_dataset = predict_dataset.select(select_indexs)
                    predict_results = trainer.predict(
                        select_predict_dataset,
                        metric_key_prefix="predict",
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    all_group_accs.append(np.array(np.concatenate(trainer.model.encoder.all_attn_weights)).mean(axis=0))
                    print(np.array(np.concatenate(trainer.model.encoder.all_attn_weights)).mean(axis=0))
                with open(os.path.join(training_args.output_dir, "group_acc.txt"), 'w') as f:
                    f.write(str(all_group_accs))
                trainer.model.encoder.is_inference = False


            predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id
            )
            metrics = predict_results.metrics
            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            trainer.log(metrics)
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
            all_metrics.update(metrics)

            with open(os.path.join("logs_and_outputs", training_args.run_name, "outputs", "task_order.txt"), 'w') as f:
                f.write(data_args.task_order)

    return results


if __name__ == "__main__":
    main()
