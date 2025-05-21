# GainLoRA

> GainLoRA (InfLoRA) with T5-Large.

## Requirements
* python 3.10.12
* pytorch 2.1.0
* transformers 4.30.2
* Datasets 2.14.6
* CUDA 12.1
* cupy 12.1.0
* deepspeed 0.11.2
* accelerate 0.24.1
* numpy 1.22.4

## Runing

### GainLoRA with InfLoRA
```
bash gen_script_superni_order1_t5_gainlora_inflora.sh your_device_id model_path_for_t5_large
bash gen_script_superni_order2_t5_gainlora_inflora.sh your_device_id model_path_for_t5_large
bash gen_script_long_order3_t5_gainlora_inflora.sh your_device_id model_path_for_t5_large
bash gen_script_long_order4_t5_gainlora_inflora.sh your_device_id model_path_for_t5_large
```
### InfLoRA
```
bash gen_script_superni_order1_t5_inflora.sh your_device_id model_path_for_t5_large
bash gen_script_superni_order2_t5_inflora.sh your_device_id model_path_for_t5_large
bash gen_script_long_order3_t5_inflora.sh your_device_id model_path_for_t5_large
bash gen_script_long_order4_t5_inflora.sh your_device_id model_path_for_t5_large
```

## Acknoledgements
The code of this repository partly relies on the following two repositories:
- [SAPT](https://github.com/circle-hit/SAPT)
- [O-LORA](https://github.com/cmnfriend/O-LoRA)