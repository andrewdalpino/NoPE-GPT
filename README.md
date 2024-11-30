# GPT

A Generative Pre-trained Transformer (GPT) trained on the Openwebtext dataset. The default implementation uses `r50k_base` BPE tokenization with a network architecture similar to OpenAI's GPT series but can easily be customized and scaled up or down to meet your needs and compute budget with some parameter adjustments. In addition, you may incorporate your own training data alongside Openwebtext for additional pre-training samples or for fine-tuning for a specific task after pre-training. It also supports PyTorch's Distributed Data Parallel (DDP) protocol for efficient training over multiple CUDA-enabled GPU clusters.

## Download the Repository
Clone the project locally using git:

```
git clone https://github.com/andrewdalpino/GPT
```

## Requirements

- [Python](https://www.python.org/) 3.10 or later
- A CUDA-enabled GPU with 12G of VRAM or more

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. I recommend using a virtual environment such as venv to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

If you'd just like to start training right away, the default settings should work on most single-GPU systems with 12G of VRAM or more.

```
python pre-train.py
```

> Note that it will take a while to download and pre-process the dataset the first time that the training script is run.

If you have a larger system you can increase the training load by increasing the capacity of the network and `batch_size` at runtime.

```
python pre-train.py --embedding_dimensions=1024 --num_hidden_layers=24 --batch_size=8
```

To distribute the training workload over a cluster of GPUs or multiple cluster nodes, use PyTorch's [torchrun](https://pytorch.org/docs/stable/elastic/run.html) extension to launch a distributed data parallel session.

```
torchrun --standalone --nnodes=1 --nproc-per-node=8 pre-train.py --batch_size=16 --gradient_accumulation_steps=32
```

> Note that when training in data-parallel mode it's important that the `gradient_accumulation_steps` divides evenly into the world size for maximum performance. For example, if we have an 8 GPU cluster, we could perform 32 gradient accumulation steps in exactly 4 passes over the network.

After training, you can generate text from the model by running the `generate.py` script from the commandline with a prompt.

```
python generate.py
```

### Pre-training Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --batch_size | 4 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 32 | int | The number of batches to pass through the network before updating the weights. |
| --samples_per_epoch | 4096 | int | The number of training samples to pass through the network every epoch. |
| --learning_rate | 5e-4 | float | The global step size taken after every gradient accumulation step. |
| --dropout | 0.1 | float | The proportion of signals to send to zero during training as regularization. |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold before stepping. |
| --num_epochs | 2145 | int | The number of epochs to train for. |
| --eval_interval | 10 | int | Evaluate the model after this many epochs on the testing set. |
| --block_size | 1024 | int | The number of tokens within the context window for every sample. |
| --embedding_dimensions | 768 | int | The dimensionality of the token embeddings. |
| --num_attention_heads | 12 | int | The number of attention heads within every block. |
| --num_hidden_layers | 12 | int | The number of attention/MLP blocks within the hidden layer of the network. |
| --checkpoint_interval | 20 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./out/checkpoint.pt" | string | The path to the checkpoint file on disk. |
| --dataset_path | "./dataset" | string | The path to the dataset files on disk. |
| --num_dataset_processes | 8 | int | The number of processes (CPUs) to use to process the dataset. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

### Fine-tuning Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --base_model_path | "./out/checkpoint.pt" | string | The path to the pre-trained model. |
| --batch_size | 4 | int | The number of samples to pass through the network at a time. |
| --mask_input | True | bool | Should we mask the input part of the sample i.e. only train on the output? |
| --gradient_accumulation_steps | 32 | int | The number of batches to pass through the network before updating the weights. |
| --learning_rate | 5e-4 | float | The global step size taken after every gradient accumulation step. |
| --rank | 8 | int | The rank of the LoRA decomposition matrices. |
| --alpha | 2.0 | float | The strength of the LoRA signal. |
| --num_epochs | 4 | int | The number of epochs to train for. |
| --eval_interval | 1 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 1 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./out/checkpoint.pt" | string | The path to the checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

### Generation Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --checkpoint_path | "./out/checkpoint.pt" | string | The path to the checkpoint file on disk. |
| --lora_path | "./out/lora.pt" | string | The path to the LoRA checkpoint. |
| --max_tokens | 500 | int | The maximum number of tokens that the model should generate per sample. |
| --temperature | 1.0 | float | The amount of regularization applied to the candidate token probabilities. |
| --top_k | 20 | int | Only sample from this many candidate tokens with the highest probabilities. |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

## References:
>- S. Rajbhandari, et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, 2020.
>- A. Vaswani, et al. Attention Is All You Need. 31st Conference on Neural Information Processing Systems, 2017.
>- A. Radford, et al. Language Models are Unsupervised Multitask Learners, OpenAI, 2019.
>- T. Brown, et al. Language Models are Few-Shot Learners. OpenAI, 2020.
>- O. Press, et. al. Using the Output Embedding to Improve Language Models.
>- J. R. Hermans, et al. Accumulated Gradient Normalization, JMLR: Workshop and Conference Proceedings, 2017.
>- R, Pascanu, et al. On the difficulty of training Recurrent Neural Networks, 2013.
>- I. Loshchilov, et al. Decoupled Weight Decay Regulaization, ILCR, 2019.
>- N. Srivastava, et al. Dropout: A Simple Way To Prevent Neural Networks from Overfitting, Journal of Machine Learning Research 15, 2014.

## License
The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).