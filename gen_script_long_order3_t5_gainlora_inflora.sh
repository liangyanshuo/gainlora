#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 20:00:00   
#SBATCH --mem 128G 
#SBATCH --gres=gpu:a100-sxm4-80gb:1  

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/gen_script_long_order3_t5_configs/yelp \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --add_instruction_replay \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/checkpoint*

sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/amazon \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_amazon \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/mnli \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_mnli \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/cb \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_cb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/copa \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_copa \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/qqp \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_qqp \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/rte \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_rte \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/imdb \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_imdb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/sst2 \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2 \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_sst2 \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/dbpedia \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_dbpedia \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/agnews \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_agnews \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/yahoo \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_yahoo \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/multirc \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/13-multirc \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_multirc \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/13-multirc/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/13-multirc/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/13-multirc/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/13-multirc/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/boolq \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/14-boolq \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_boolq \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/14-boolq/checkpoint*
   
sleep 5


CUDA_VISIBLE_DEVICES=$1 python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $2 \
   --load_checkpoint_from logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/14-boolq/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/1-yelp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/2-amazon/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/3-mnli/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/4-cb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/5-copa/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/6-qqp/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/7-rte/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/8-imdb/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/9-sst2/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/10-dbpedia/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/11-agnews/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/12-yahoo/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/13-multirc/saved_weights,logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/14-boolq/saved_weights \
   --previous_prompt_key_path logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/14-boolq/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/gen_script_long_order3_t5_configs/wic \
   --output_dir logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/15-wic \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name gen_script_long_order3_t5_gainlora_inflora \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_wic \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --kl_ratio 0.1 \
   --attn_temperature 1 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 \
   --transthreshold 0.995

rm -rf logs_and_outputs/gen_script_long_order3_t5_gainlora_inflora/outputs/15-wic/checkpoint*
   
sleep 5

CUDA_VISIBLE_DEVICES=$1 python score.py gen_script_long_order3_t5_gainlora_inflora gen_script_long_order3_t5_gainlora_inflora



