import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
import numpy as np

from cl_collator import SUPPORTED_DECODER_MODELS, check_model
from cl_dataset import ANSWER_PREFIX
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupy import fromDlpack
import ipdb

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:
            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions

def create_memory_replay_generators(task, task_list, replay_data_dict, split='train_mem'): # creating previous tasks memory buffers
    print('Creating generators for previous tasks ...')
    tasks_to_generators = {}
    curr_task_num = task_list.index(task)
    for idx in np.arange(curr_task_num):
        prev_task = task_list[idx]
        tasks_to_generators[prev_task] = iter(replay_data_dict[prev_task])
    return tasks_to_generators

class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class InfLoRA_Trainer(Seq2SeqTrainer):

    def __init__(self, model, args, train_dataset, cur_task_id, task_order, data_collator_replay=None, replay_dataset_dict=None, replay_label_dict=None, eval_dataset=None, tokenizer=None, data_collator=None, compute_metrics=None, callbacks=None):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics, callbacks=callbacks)

        self.data_collator_replay = data_collator_replay
        self.replay_dataset_dict = replay_dataset_dict
        self.replay_label_dict = replay_label_dict
        self.task_order = task_order
        self.cur_task_id = cur_task_id

        if self.args.data_replay_freq != -1:
            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            self.replay_dataloader_dict = {}
            generator = torch.Generator()
            generator.manual_seed(seed)
            if replay_dataset_dict is not None:
                for dataset_name, dataset in self.replay_dataset_dict.items():
                    train_sampler = RandomSampler(dataset, generator=generator)
                    self.replay_dataloader_dict[dataset_name] = DataLoader(
                        dataset,
                        batch_size=self._train_batch_size,
                        sampler=train_sampler,
                        collate_fn=self.data_collator_replay,
                        drop_last=self.args.dataloader_drop_last,
                        num_workers=self.args.dataloader_num_workers,
                        pin_memory=False,
                        worker_init_fn=seed_worker)
            self.replay_iterator_dict = create_memory_replay_generators(task_order[cur_task_id], task_order, self.replay_dataloader_dict)
    
    def load_previous_reg_matrix(self):
        paths = self.args.output_dir.split('/')
        log_path = ""
        for path in paths[:-1]:
            log_path = os.path.join(log_path, path)
        print(log_path)
        local_dir = paths[-1]

        all_dirs = os.listdir(log_path)
        reg_matrix = []
        for all_dir in all_dirs:
            if not os.path.isdir(os.path.join(log_path, all_dir)): continue
            if eval(all_dir.split('-')[0]) == eval(local_dir.split('-')[0])-1: 
                i = 0
                for module in self.model.modules():
                    if hasattr(module, 'get_feature'):
                        reg_matrix.append(torch.load(os.path.join(os.path.join(log_path, all_dir), "reg_{}.pt".format(i))))
                        i += 1
                # reg_matrixs.append(reg_matrix)
                print(os.path.join(log_path, all_dir))
                print(len(reg_matrix))
                break
        return reg_matrix, eval(local_dir.split('-')[0])-1

    def get_reg_matrix(self):
        # if self.args.lamda_1 <= 1e-6:
        #     return
        self.feature_list, self._cur_task = self.load_previous_reg_matrix()
        if len(self.feature_list) == 0:
            return
        # import ipdb
        # ipdb.set_trace()
        self.feature_mat, i = [], 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                feature_mat = {}
                for index in self.feature_list[i].keys():
                    feature_mat[index] = torch.mm(self.feature_list[i][index], self.feature_list[i][index].T).cuda()
                self.feature_mat.append(feature_mat)
                # print(feature_mat)
                # print(np.sum(self.feature_list[i]**2,axis=0))
                # exit()
                for index in self.feature_list[i].keys():
                    # pre = deepcopy(module.loranew_A[module.active_adapter].weight.data[:,index*module.step:(index+1)*module.step])
                    # print(index*module.step, (index+1)*module.step)
                    # print(feature_mat[index])
                    # print()
                    # print(torch.mm(module.loranew_A[module.active_adapter].weight[:,index*module.step:(index+1)*module.step].data,feature_mat[index].cpu()))
                    # print()
                    module.lora_q.lora_A.data[:,index*module.step:(index+1)*module.step].copy_(module.lora_q.lora_A.data[:,index*module.step:(index+1)*module.step] - torch.mm(module.lora_q.lora_A.data[:,index*module.step:(index+1)*module.step], feature_mat[index]))
                    module.lora_v.lora_A.data[:,index*module.step:(index+1)*module.step].copy_(module.lora_v.lora_A.data[:,index*module.step:(index+1)*module.step] - torch.mm(module.lora_v.lora_A.data[:,index*module.step:(index+1)*module.step], feature_mat[index]))
                module.lora_q.lora_A.data /= (math.sqrt(3) * module.lora_q.lora_A.data.norm(dim=1,keepdim=True))
                module.lora_v.lora_A.data /= (math.sqrt(3) * module.lora_v.lora_A.data.norm(dim=1,keepdim=True))
                    # print(pre - module.loranew_A[module.active_adapter].weight.data[:,index*module.step:(index+1)*module.step])
                    # print()
                # for index in self.feature_list[i].keys():
                #     print(torch.mm(module.loranew_A[module.active_adapter].weight[:,index*module.step:(index+1)*module.step].data,feature_mat[index].cpu()))
                #     print()
                # exit()
                i += 1
        return


    def get_repsentation(self):
        # if self.args.lamda_1 <= 1e-6:
        #     return
        self.feature_list, self._cur_task = self.load_previous_reg_matrix()

        train_dataloader = self.get_train_dataloader()

        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(1998)
        elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
            train_dataloader.dataset.set_epoch(1998)

        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                module.get_feature=True
                module.stage = 0

        # for name, module in self.model.named_modules():
        #     if hasattr(module, 'get_feature'):
        #         print(module.get_feature)
        #         break

        print('begin get representation')
        with torch.no_grad():
            for step, inputs in enumerate(train_dataloader):
                inputs = self._prepare_inputs(inputs)
                if self.label_smoother is not None and "labels" in inputs:
                    labels = inputs.pop("labels")
                else:
                    labels = None
                # del inputs['task_ids']
                outputs = self.model(**inputs)
                if step > 1000: break
        print('end get representation')


        mat_list = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                # # 创建一个 CPU 上的张量，在每个进程中填充不同的值
                # rank = dist.get_rank()
                # local_tensor = torch.tensor([rank + 1])  # 为每个进程创建不同的值

                # print(module.matrix, module.weight.device)
                merged_tensor = {}
                for index in range(module.index):
                    merged_tensor[index] = module.matrix[index].cuda().float()

                mat_list.append(merged_tensor)

                module.get_feature=False
                module.stage = 0

        # U, S, V = torch.linalg.svd(merged_tensor)

        total_sessions = 15
        threshold = (1.0 - self.args.threshold)*self._cur_task/total_sessions + self.args.threshold
        # threshold = self.args.threshold
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                feature = {}
                for index in activation.keys():
                    U,S,Vh = cp.linalg.svd(fromDlpack(to_dlpack(activation[index])), full_matrices=False)
                    U = from_dlpack(U.toDlpack())
                    S = from_dlpack(S.toDlpack())
                    # criteria (Eq-5)
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = torch.sum(torch.cumsum(sval_ratio, dim=0)<threshold) #+1  
                    feature[index] = U[:,0:max(r,1)]
                self.feature_list.append(feature)
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                feature = {}
                for index in activation.keys():
                    U1,S1,Vh1=cp.linalg.svd(fromDlpack(to_dlpack(activation[index])), full_matrices=False)
                    # S1 = from_dlpack(S1.toDlpack())
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = fromDlpack(to_dlpack(activation[index])) - cp.dot(cp.dot(fromDlpack(to_dlpack(self.feature_list[i][index])),fromDlpack(to_dlpack(self.feature_list[i][index].T))),fromDlpack(to_dlpack(activation[index])))
                    U,S,Vh = cp.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
                
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=cp.hstack((fromDlpack(to_dlpack(self.feature_list[i][index])),U[:,0:r]))  

                    # import ipdb
                    # ipdb.set_trace()
                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_list[i][index]=from_dlpack(Ui[:,0:Ui.shape[0]].toDlpack())
                    else:
                        self.feature_list[i][index]=from_dlpack(Ui.toDlpack())
    
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            for index in range(self.args.chunk):
                print ('Layer {} Index {} : {}/{}'.format(i+1, index+1, self.feature_list[i][index].shape[1], self.feature_list[i][index].shape[0]))
        print('-'*40)  

        for i in range(len(self.feature_list)):
            torch.save(self.feature_list[i], os.path.join(self.args.output_dir, 'reg_{}.pt'.format(i)))


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.is_deepspeed_enabled:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.is_deepspeed_enabled:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        if self.state.global_step > self.args.replay_after_n_epoch*self.args.step_per_epoch and self.args.data_replay_freq != -1 and self.state.global_step % self.args.data_replay_freq == 0:
            for item in self.replay_iterator_dict.keys():
                generator_mem1 = self.replay_iterator_dict[item]
                try:
                    # Samples the batch
                    b = next(generator_mem1)
                except StopIteration:
                    generator_mem1 = iter(self.replay_dataloader_dict[item])

                    self.replay_iterator_dict[item] = generator_mem1
                    b = next(generator_mem1)

                replay_task_id = self.task_order.index(item)
                b["replay_labels"] = self.replay_label_dict[self.task_order[replay_task_id]]
                
                replay_inputs = self._prepare_inputs(b)
                with self.compute_loss_context_manager():
                    kl_loss = self.args.kl_ratio * self.model.memory_replay(replay_inputs["input_ids"], replay_inputs["replay_labels"])

                if self.args.n_gpu > 1:
                    kl_loss = kl_loss.mean()  # mean() to average on multi-gpu parallel trainin
        
                if self.do_grad_scaling:
                    self.scaler.scale(kl_loss).backward()
                elif self.use_apex:
                    with amp.scale_loss(kl_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif self.is_deepspeed_enabled:
                    self.accelerator.backward(kl_loss)
                else:
                    kl_loss.backward()

        return loss.detach()
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        if self.optimizer is None:
            if self.args.attn_lr == 0:
                print("Using Same Learning Rate for All Modules")
                decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                print("Using Different Learning Rates for Different Modules")
                decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                
                param_no_decay = [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad]
                
                resett_param_with_decay = [p for n, p in opt_model.named_parameters() if "trans_input" in n and n in decay_parameters and p.requires_grad]
                other_param_with_decay = [p for n, p in opt_model.named_parameters() if "trans_input" not in n and n in decay_parameters and p.requires_grad]
                optimizer_grouped_parameters = [
                    {
                        "params": other_param_with_decay,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate
                    },
                    {
                        "params": resett_param_with_decay,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.attn_lr
                    },
                    {
                        "params": param_no_decay,
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.is_deepspeed_enabled:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # gen_kwargs = self._gen_kwargs
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            # T5 generation config
            gen_kwargs = {
                "max_new_tokens": 50,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "decoder_start_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 0,
            }
            gen_kwargs["synced_gpus"] = False
        else:
            if inputs.get("input_ids_wo_label", None) is not None:
                # LLaMA-2 generation config
                gen_kwargs = {
                    "bos_token_id": 1,
                    "max_new_tokens": 50,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "repetition_penalty": 1.0,
                    "eos_token_id": 2,
                    "pad_token_id": 1,
                }
            else:
                # T5 generation config
                gen_kwargs = {
                    "max_new_tokens": 50,
                    "num_beams": 1,
                    "repetition_penalty": 1.0,
                    "decoder_start_token_id": 0,
                    "eos_token_id": 1,
                    "pad_token_id": 0,
                }
                
            gen_kwargs["synced_gpus"] = False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
            
            generated_tokens = self.model.generate(
                input_ids=generation_inputs, 
                generation_config=generation_config,
            )
        else:
            generation_inputs = inputs[self.model.main_input_name]

            if inputs.get("input_ids_wo_label", None) is not None:
                generated_tokens = self.model.generate(
                    input_ids=generation_inputs,
                    input_ids_wo_label=inputs["input_ids_wo_label"],
                    generation_config=generation_config,
                )
            
            else:
                generated_tokens = self.model.generate(
                    input_ids=generation_inputs,
                    generation_config=generation_config,
                )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
