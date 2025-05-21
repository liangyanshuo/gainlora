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


class GainLoRA_InfLoRA_Trainer(Seq2SeqTrainer):

    def __init__(self, model, args, train_dataset, cur_task_id, task_order, data_collator_replay=None, replay_dataset_dict=None, replay_label_dict=None, eval_dataset=None, tokenizer=None, data_collator=None, compute_metrics=None, callbacks=None):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics, callbacks=callbacks)

        self.data_collator_replay = data_collator_replay
        self.replay_dataset_dict = replay_dataset_dict
        self.replay_label_dict = replay_label_dict
        self.task_order = task_order
        self.cur_task_id = cur_task_id

    
    def load_previous_reg_matrix(self):
        paths = self.args.output_dir.split('/')
        log_path = ""
        for path in paths[:-1]:
            log_path = os.path.join(log_path, path)
        print(log_path)
        local_dir = paths[-1]

        all_dirs = os.listdir(log_path)
        reg_matrix, reg_trans_matrix = [], []
        for all_dir in all_dirs:
            if not os.path.isdir(os.path.join(log_path, all_dir)): continue
            if eval(all_dir.split('-')[0]) == eval(local_dir.split('-')[0])-1: 
                i = 0
                for module in self.model.modules():
                    if hasattr(module, 'get_feature'):
                        reg_matrix.append(torch.load(os.path.join(os.path.join(log_path, all_dir), "reg_{}.pt".format(i))))
                        i += 1
                reg_trans_matrix.append(torch.load(os.path.join(os.path.join(log_path, all_dir, 'trans_input'), "reg_0.pt")))
                reg_trans_matrix.append(torch.load(os.path.join(os.path.join(log_path, all_dir, 'trans_input'), "reg_1.pt")))
                reg_trans_matrix.append(torch.load(os.path.join(os.path.join(log_path, all_dir, 'trans_input'), "reg_2.pt")))
                # for module in self.model.modules():
                #     if hasattr(module, 'get_trans_feature'):
                #         reg_matrix.append(torch.load(os.path.join(os.path.join(log_path, all_dir, 'trans_input'), "reg_{}.pt".format(i))))
                #         i += 1
                # reg_matrixs.append(reg_matrix)
                print(os.path.join(log_path, all_dir))
                print(len(reg_matrix))
                break
        return reg_matrix, reg_trans_matrix, eval(local_dir.split('-')[0])-1


    def get_reg_matrix(self):
        # if self.args.lamda_1 <= 1e-6:
        #     return
        self.feature_list, self.feature_trans_list, self._cur_task = self.load_previous_reg_matrix()
        # import ipdb

        train_dataloader = self.get_train_dataloader()
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(1998)
        elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
            train_dataloader.dataset.set_epoch(1998)
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                module.get_feature=True
                module.stage = 0
        self.model.encoder.get_trans_feature = True
        self.model.encoder.stage_trans = 0

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

        if len(self.feature_list) == 0:
            i = 0
            for module in self.model.modules():
                if hasattr(module, 'get_feature'):
                    for index in module.matrix.keys():
                        cur_matrix = module.matrix[index]
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_q.lora_A.data[:,index*module.step:(index+1)*module.step].copy_(U[:,:module.lora_q.r].T)
                        module.lora_v.lora_A.data[:,index*module.step:(index+1)*module.step].copy_(U[:,:module.lora_v.r].T)
                        module.matrix[index].zero_()
                        module.n_matrix[index] = 0
                    module.lora_q.lora_A.data /= math.sqrt(module.chunk*3)
                    module.lora_v.lora_A.data /= math.sqrt(module.chunk*3)
                    i += 1
                    module.get_feature=False
                    module.stage = 0

            module = self.model.encoder
            pre_norm = module.prompt_key.detach().norm()
            for index in module.matrix_trans_3.keys():
                cur_trans_matrix = module.matrix_trans_3[index]
                U, S, V = torch.linalg.svd(cur_trans_matrix)
                module.prompt_key.data[:,index*module.step:(index+1)*module.step].copy_(U[:,:1].T)
                # ipdb.set_trace()
                module.matrix_trans_1[index].zero_()
                module.matrix_trans_3[index].zero_()
                module.n_trans_matrix[index] = 0
            module.matrix_trans_2.zero_()
            module.prompt_key.data /= math.sqrt(module.chunk_trans)
            module.prompt_key.data *= pre_norm
            module.get_trans_feature=False
            module.stage_trans=0
        else:
            self.feature_mat, i = [], 0
            for module in self.model.modules():
                if hasattr(module, 'get_feature'):
                    feature_mat = {}
                    for index in self.feature_list[i].keys():
                        feature_mat[index] = torch.mm(self.feature_list[i][index], self.feature_list[i][index].T)
                    self.feature_mat.append(feature_mat)
                    i += 1

            self.feature_trans_mat = []
            feature_trans_mat = {}
            for index in self.feature_trans_list[0].keys():
                feature_trans_mat[index] = torch.mm(self.feature_trans_list[0][index], self.feature_trans_list[0][index].T)
            self.feature_trans_mat.append(feature_trans_mat)
            self.feature_trans_mat.append(torch.mm(self.feature_trans_list[1], self.feature_trans_list[1].T))
            feature_trans_mat = {}
            try:
                for index in self.feature_trans_list[2].keys():
                    feature_trans_mat[index] = torch.mm(self.feature_trans_list[2][index], self.feature_trans_list[2][index].T)
            except:
                ipdb.set_trace()
                raise Exception
            # ipdb.set_trace()
            self.feature_trans_mat.append(feature_trans_mat)

            i = 0
            for module in self.model.modules():
                if hasattr(module, 'get_feature'):
                    for index in module.matrix.keys():
                        cur_matrix = module.matrix[index]
                        cur_matrix = cur_matrix - torch.mm(self.feature_mat[i][index],cur_matrix)
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_q.lora_A.data[:,index*module.step:(index+1)*module.step].copy_(U[:,:module.lora_q.r].T)
                        module.lora_v.lora_A.data[:,index*module.step:(index+1)*module.step].copy_(U[:,:module.lora_v.r].T)
                        module.matrix[index].zero_()
                        module.n_matrix[index] = 0
                    module.lora_q.lora_A.data /= math.sqrt(module.chunk*3)
                    module.lora_v.lora_A.data /= math.sqrt(module.chunk*3)
                    i += 1
                    module.get_feature=False
                    module.stage = 0

            module = self.model.encoder
            pre_norm = module.prompt_key.detach().norm()
            for index in module.matrix_trans_3.keys():
                cur_trans_matrix = module.matrix_trans_3[index]
                try:
                    cur_trans_matrix = cur_trans_matrix - torch.mm(self.feature_trans_mat[2][index],cur_trans_matrix)
                except:
                    ipdb.set_trace()
                    raise Exception
                U, S, V = torch.linalg.svd(cur_trans_matrix)
                module.prompt_key.data[:,index*module.step:(index+1)*module.step].copy_(U[:,:1].T)
                module.matrix_trans_1[index].zero_()
                module.matrix_trans_3[index].zero_()
                module.n_trans_matrix[index] = 0
            module.matrix_trans_2.zero_()
            module.prompt_key.data /= math.sqrt(module.chunk_trans)
            module.prompt_key.data *= pre_norm
            module.get_trans_feature=False
            module.stage_trans=0
        return

    def get_repsentation(self):
        # if self.args.lamda_1 <= 1e-6:
        #     return
        self.feature_list, self.feature_trans_list, self._cur_task = self.load_previous_reg_matrix()
        # ipdb.set_trace()

        train_dataloader = self.get_train_dataloader()

        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(1998)
        elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
            train_dataloader.dataset.set_epoch(1998)

        for name, module in self.model.named_modules():
            if hasattr(module, 'get_feature'):
                module.get_feature=True
                module.stage = 0
        self.model.encoder.get_trans_feature = True
        self.model.encoder.stage_trans = 0

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


        mat_list, mat_trans_list = [], []
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
        
        merged_trans_tensor = {}
        for index in range(self.model.encoder.index):
            merged_trans_tensor[index] = self.model.encoder.matrix_trans_1[index].cuda().float()
        mat_trans_list.append(merged_trans_tensor)
        mat_trans_list.append(self.model.encoder.matrix_trans_2.cuda().float())
        merged_trans_tensor = {}
        for index in range(self.model.encoder.index):
            merged_trans_tensor[index] = self.model.encoder.matrix_trans_3[index].cuda().float()
        mat_trans_list.append(merged_trans_tensor)
        self.model.encoder.get_trans_feature = False
        self.model.encoder.stage_trans = 0

        # U, S, V = torch.linalg.svd(merged_tensor)

        total_sessions = 15
        threshold = (1.0 - self.args.threshold)*self._cur_task/total_sessions + self.args.threshold
        if 'long' in self.args.output_dir:
            transthreshold = (1.0 - self.args.transthreshold)*self._cur_task/total_sessions + self.args.transthreshold
            # transthreshold = self.args.transthreshold
        else:
            transthreshold = (1.0 - self.args.transthreshold)*self._cur_task/total_sessions + self.args.transthreshold
        # threshold = self.args.threshold
        print ('Threshold: ', threshold, transthreshold) 
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

            for i in range(3):
                if i == 1: continue
                activation_trans = mat_trans_list[i]
                feature_trans = {}
                for index in activation_trans.keys():
                    U,S,Vh = cp.linalg.svd(fromDlpack(to_dlpack(activation_trans[index])), full_matrices=False)
                    U = from_dlpack(U.toDlpack())
                    S = from_dlpack(S.toDlpack())
                    # criteria (Eq-5)
                    sval_total = (S**2).sum()
                    sval_ratio = (S**2)/sval_total
                    r = torch.sum(torch.cumsum(sval_ratio, dim=0)<transthreshold) #+1  
                    feature_trans[index] = U[:,0:max(r,1)]
                self.feature_trans_list.append(feature_trans)

            activation_trans = mat_trans_list[1]
            U,S,Vh = cp.linalg.svd(fromDlpack(to_dlpack(activation_trans)), full_matrices=False)
            U = from_dlpack(U.toDlpack())
            S = from_dlpack(S.toDlpack())
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = torch.sum(torch.cumsum(sval_ratio, dim=0)<transthreshold) #+1  
            feature_trans = U[:,0:max(r,1)]
            self.feature_trans_list = self.feature_trans_list[:1] + [feature_trans] + self.feature_trans_list[1:]
            # ipdb.set_trace()

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

            # ipdb.set_trace()
            for i in range(3):
                if i == 1: continue
                # ipdb.set_trace()
                activation_trans = mat_trans_list[i]
                feature_trans = {}
                for index in activation_trans.keys():
                    U1,S1,Vh1=cp.linalg.svd(fromDlpack(to_dlpack(activation_trans[index])), full_matrices=False)
                    # S1 = from_dlpack(S1.toDlpack())
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = fromDlpack(to_dlpack(activation_trans[index])) - cp.dot(cp.dot(fromDlpack(to_dlpack(self.feature_trans_list[i][index])),fromDlpack(to_dlpack(self.feature_trans_list[i][index].T))),fromDlpack(to_dlpack(activation_trans[index])))
                    U,S,Vh = cp.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
                
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < transthreshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=cp.hstack((fromDlpack(to_dlpack(self.feature_trans_list[i][index])),U[:,0:r]))  

                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_trans_list[i][index]=from_dlpack(Ui[:,0:Ui.shape[0]].toDlpack())
                    else:
                        self.feature_trans_list[i][index]=from_dlpack(Ui.toDlpack())

            activation_trans = mat_trans_list[1]
            feature_trans = {}
            U1,S1,Vh1=cp.linalg.svd(fromDlpack(to_dlpack(activation_trans)), full_matrices=False)
            # S1 = from_dlpack(S1.toDlpack())
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = fromDlpack(to_dlpack(activation_trans)) - cp.dot(cp.dot(fromDlpack(to_dlpack(self.feature_trans_list[1])),fromDlpack(to_dlpack(self.feature_trans_list[1].T))),fromDlpack(to_dlpack(activation_trans)))
            U,S,Vh = cp.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
        
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < transthreshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(1+1)) 
            else:
                # update GPM
                Ui=cp.hstack((fromDlpack(to_dlpack(self.feature_trans_list[1])),U[:,0:r]))  

                # import ipdb
                # ipdb.set_trace()
                if Ui.shape[1] > Ui.shape[0]:
                    self.feature_trans_list[1]=from_dlpack(Ui[:,0:Ui.shape[0]].toDlpack())
                else:
                    self.feature_trans_list[1]=from_dlpack(Ui.toDlpack())

            
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            for index in range(self.args.chunk):
                print ('Layer {} Index {} : {}/{}'.format(i+1, index+1, self.feature_list[i][index].shape[1], self.feature_list[i][index].shape[0]))
        print('-'*40)  

        for i in range(len(self.feature_list)):
            torch.save(self.feature_list[i], os.path.join(self.args.output_dir, 'reg_{}.pt'.format(i)))
        # ipdb.set_trace()

        os.makedirs(os.path.join(self.args.output_dir, 'trans_input'), exist_ok=True)
        for i in range(len(self.feature_trans_list)):
            torch.save(self.feature_trans_list[i], os.path.join(self.args.output_dir, 'trans_input', 'reg_{}.pt'.format(i)))


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




    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):

                    if self._cur_task:
                        from copy import deepcopy
                        # old_params_q, old_params_v, num_train_modules = [], [], []
                        old_trans_input_0 = deepcopy(self.model.encoder.trans_input[0].weight.detach())
                        old_trans_input_1 = deepcopy(self.model.encoder.trans_input[2].weight.detach())
                        old_prompt_key = deepcopy(self.model.encoder.prompt_key.detach())

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()


                    if self._cur_task:

                        new_trans_input_0 = deepcopy(self.model.encoder.trans_input[0].weight.detach())
                        new_trans_input_1 = deepcopy(self.model.encoder.trans_input[2].weight.detach())
                        new_trans_input_0norm = new_trans_input_0.norm(dim=1, keepdim=True)
                        new_trans_input_1norm = new_trans_input_1.norm(dim=1, keepdim=True)

                        new_prompt_key = deepcopy(self.model.encoder.prompt_key.detach())
                        new_prompt_key_norm = new_prompt_key.norm(dim=1, keepdim=True)
                        for index in self.feature_trans_mat[0].keys():
                            new_trans_input_0[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step] = self.model.encoder.trans_input[0].weight.detach()[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step] - torch.mm(self.model.encoder.trans_input[0].weight.detach()[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step]-old_trans_input_0[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step], self.feature_trans_mat[0][index])
                            new_prompt_key[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step] = self.model.encoder.prompt_key.detach()[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step] - torch.mm(self.model.encoder.prompt_key.detach()[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step]-old_prompt_key[:,index*self.model.encoder.step:(index+1)*self.model.encoder.step], self.feature_trans_mat[2][index])
                        new_trans_input_1 = self.model.encoder.trans_input[2].weight.detach() - torch.mm(self.model.encoder.trans_input[2].weight.detach()-old_trans_input_1, self.feature_trans_mat[1])

                        new_trans_input_0 = new_trans_input_0*new_trans_input_0norm / new_trans_input_0.norm(dim=1, keepdim=True)
                        new_trans_input_1 = new_trans_input_1*new_trans_input_1norm / new_trans_input_1.norm(dim=1, keepdim=True)
                        new_prompt_key = new_prompt_key*new_prompt_key_norm / new_prompt_key.norm(dim=1, keepdim=True)

                        self.model.encoder.trans_input[0].weight.data.copy_(new_trans_input_0)
                        self.model.encoder.trans_input[2].weight.data.copy_(new_trans_input_1)
                        self.model.encoder.prompt_key.data.copy_(new_prompt_key)

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


