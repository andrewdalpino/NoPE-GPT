---
license: apache-2.0
datasets:
- HuggingFaceFW/fineweb
- HuggingFaceTB/smoltalk
library_name: pytorch
language:
- en
metrics:
- perplexity
pipeline_tag: text-generation
tags:
- NoPE
---
# LightGPT

LightGPT is a lightweight generative pretrained Transformer (GPT) language model for the people! Built using PyTorch and trained on the Fineweb and SmolTalk datasets, LightGPT can answer questions, follow instructions, summarize documents, chat, and more. Best of all, the model weights *and* code are fully open-source for you to customize, improve upon, and share with the world.

## Features

- **No positional embeddings**: LightGPT aims to be a more parsimonious model by completely removing positional embeddings from the architecture. This allows for a variable context length without complex model surgery. Despite having no positional embeddings (NoPE), LightGPT performs better at context length generalization than the best relative embeddings (ALiBi, RoPE, T5) offering good performance even at 2X of the trained context length.

- **Low Memory Utilization**: LightGPT lets you progressively employ training-time memory optimizations such as fully-sharded data-parallel (FSDP), activation checkpointing, mixed precision, and low-memory optimizer updates that allow you to train larger models on smaller hardware.

- **Fully Open-source**: Unlike closed-source LLMs, LightGPT provides both the model weights *and* the source code to train, fine-tune, export, and generate text from the model using your own hardware. With the help of the open-source software community, we aim to democratize access to AI and continually improve the models.

## Suggested Pretraining Configurations

Below is a table of some suggested pretraining configurations but feel free to experiment with settings on your own. See the `model_sizing.ipynb` notebook to estimate the memory and compute requirements for your model configuration.

| Name | Vocab. Size | Embedding Dim. | Attn. Heads | Layers | Parameters | Training Tokens |
|---|---|---|---|---|---|---|
| Small | 50,257 | 1024 | 16 | 24 | 353M | 7B |
| Medium | 50,257 | 2048 | 32 | 32 | 1.7B | 34B |
| Large | 100,275 | 4096 | 64 | 32 | 6.8B | 132B |
| X-large | 100,275 | 4096 | 64 | 64 | 13B | 262B |
| XX-large | 200,017 | 8192 | 128 | 64 | 53B | 1T |
| XXX-large | 200,017 | 8192 | 128 | 128 | 105B | 2T |

We typically recommend a training `block size` (also referred to as context length) of between 1024 to 4096 for standard models and 4096 or higher for long-context applications such as conversational chatbots, retrieval augmented generation, and chain-of-thought prompting.

**Note**: LightGPT can be trained using variable block sizes since the architecture does not depend on any discrete positional embeddings. This flexibility allows you to gradually extend the context length.

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Pretraining

For the pretraining corpus we use the Fineweb dataset which consists of about 15T high-quality tokens gathered from the worldwide web. The dataset has been split into 3 subsets (10BT, 100BT, and 350BT versions) for training smaller models. If you'd like to start training right away, the default settings should work on most single-GPU systems with 12G of VRAM or more.

```
python pretrain.py
```

**Note** that it will take a while to download and pre-process the dataset the first time that the training script is run.

To customize the default "Small" architecture you can adjust the `embedding_dimensions`, `num_attention_heads`, `num_hidden_layers`, and `feed_forward_ratio` arguments of the pretraining script. 

```
python pretrain.py --embedding_dimensions=4096 --num_attention_heads=64 --num_hidden_layers=48 --feed_forward_ratio=4
```

You can also adjust the `batch_size`, `learning_rate`, and `gradient_accumulation_steps` to suite your training setup.

```
python pretrain.py --batch_size=32 --learning_rate=0.01 --gradient_accumulation_steps=128
```

For distributed training, use PyTorch's [torchrun](https://pytorch.org/docs/stable/elastic/run.html) extension to launch a distributed data parallel (DDP) session. The example below is for executing the training script on a single node with 8 individual GPUs.

```
torchrun --standalone --nnodes=1 --nproc-per-node=8 pretrain.py --batch_size=16 --gradient_accumulation_steps=128
```

**Note** that when training in data-parallel mode it's important that the `gradient_accumulation_steps` divides evenly into the world size for maximum performance. For example, if we have an 8 GPU cluster, we could perform 32 gradient accumulation steps in exactly 4 passes over the network.

### Pretraining Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --dataset_subset | "sample-10BT" | str | The subset of the Fineweb dataset to train on. Options are `sample-10BT`, `sample-100BT`, and `sample-350BT`. Set to `None` to train on the full 15T token dataset. |
| --token_encoding | "r50k_base" | str | The Tiktoken encoding scheme to use when tokenizing the dataset. Options include `r50k_base`, `p50k_base`, `cl100k_base`, and `o200k_base`. |
| --dataset_path | "./datasets" | str | The path to the preprocessed dataset files on disk. |
| --num_dataset_processes | 8 | int | The number of processes (CPUs) to use to process the dataset. |
| --batch_size | 1 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 128 | int | The number of batches to pass through the network before updating the weights. |
| --tokens_per_sample | 1024 | int | The number of tokens to pack into a single training sequence. This is sometimes called the context length or block size. |
| --samples_per_epoch | 4096 | int | The number of training samples to pass through the network every epoch. |
| --num_epochs | 1686 | int | The number of epochs to train for. |
| --learning_rate | 1e-2 | float | The learning rate of the Adafactor optimizer. |
| --rms_decay | -0.8 | float | The decay rate of the RMS coefficient of the Adafactor optimizer. |
| --low_memory_optimizer | False | bool | Should the optimizer reduce its memory consumption in exchange for a slightly slower runtime? |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold before stepping. |
| --eval_interval | 10 | int | Evaluate the model after this many epochs on the testing set. |
| --embedding_dimensions | 1024 | int | The dimensionality of the token embeddings. |
| --num_attention_heads | 16 | int | The number of attention heads within every block. |
| --num_hidden_layers | 24 | int | The number of attention/MLP blocks within the hidden layer of the network. |
| --feed_forward_ratio | 4 | (1, 2, 4) | The ratio of hidden neurons to embedding dimensions in the MLP layers of the network. |
| --dropout | 0.1 | float | The proportion of signals to send to zero during training as regularization. |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will drastically reduce memory utilization during training at the cost of recomputing the forward pass. |
| --ddp_sharding_level | 2 | int | The level of sharding to use for DDP training. Options are 2 or 3 for partial and full sharding respectively, or 0 for no sharding. |
| --checkpoint_interval | 20 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs/pretrain" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cuda" | str | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

### Training Dashboard

We use TensorBoard to capture and display pretraining events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```
tensorboard --logdir=./runs
```

Then navigate to the dashboard using your favorite web browser.

## Instruction-tuning

### Instruction-tuning Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --base_model_path | "./checkpoints/checkpoint.pt" | string | The path to the base checkpoint on disk. |
| --max_tokens_per_sample | 1024 | int | The maximum number of tokens to pack into a single training sequence. |
| --mask_input | False | bool | Should we mask the input part of the training sequences i.e. only train on the supervised  output? |
| --batch_size | 1 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 64 | int | The number of batches to pass through the network before updating the weights. |
| --learning_rate | 5e-4 | float | The learning rate of the Adafactor optimizer. |
| --rms_decay | -0.8 | float | The decay rate of the RMS coefficient of the Adafactor optimizer. |
| --optimizer_low_memory | False | bool | Should the optimizer reduce its memory consumption in exchange for a slightly slower runtime? |
| --rank | 8 | int | The rank of the LoRA decomposition matrices. |
| --alpha | 1.0 | float | The strength of the LoRA signal. |
| --dropout | 0.05 | float | The proportion of signals to send to zero during training as regularization. |
| --num_epochs | 4 | int | The number of epochs to train for. |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will reduce drastically memory utilization during training at the cost of needing to recompute the forward pass. |
| --eval_interval | 1 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 1 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/lora_instruction.pt" | string | The path to the LoRA checkpoint. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs/instruction-tune" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

## Text Generation

After training, you can generate text from the model by running the `generate.py` script from the commandline. This inference script samples tokens from the model one at a time conditioned on a prompt and any previously generated tokens, together referred to as the context window. In the example below we are choosing to only sample from the `top_k` predicted tokens that have at least `top_p` cumulative probability mass when ordered descending by predicted probability.

```
python generate.py --top_k=500 --top_p=0.9
```

### Generation Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --checkpoint_path | "./checkpoints/checkpoint.pt" | string | The path to the base checkpoint file on disk. |
| --lora_path | None | string | The path to the LoRA checkpoint. |
| --max_tokens | 1000 | int | The maximum number of tokens that the model should generate per sample. |
| --context_length | 1024 | int | The number of tokens to keep within the context window of the current prediction. |
| --temperature | 1.0 | float | The amount of regularization applied to the candidate token probabilities. |
| --top_k | 500 | int | Only sample from this many candidate tokens with the highest probabilities. |
| --top_p | 0.9 | float | Of the `top_k` tokens, drop all but the `top_p` portion of the cumulative probability distribution. |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

We also provide a script that samples entire sequences rather than single tokens independently which we call `beam_search.py`. Beam Search maintains a list of the top `beam_width` sequence candidates and outputs the top `num_candidates` completed sequences with the highest overall priority. It is a form of greedy search that works well for some things like text summarization and translation but often results in less natural responses as natural language follows a more stochastic process.

```
python beam_search.py --beam_width=16 --num_candidates=3
```

### Beam Search Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --checkpoint_path | "./checkpoints/checkpoint.pt" | string | The path to the base checkpoint file on disk. |
| --lora_path | None | string | The path to the LoRA checkpoint. |
| --max_tokens | 100 | int | The maximum number of tokens that the model should generate per sample. |
| --context_length | 1024 | int | The number of tokens to keep within the context window of the current prediction. |
| --num_candidates | 3 | int | The number of candidate sequences to output. |
| --beam_width | 16 | int | The number of candidate sequences to keep track of during search. |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

## References:
>- G. Penedo, et al. The FineWeb Datasts: Decanting the Web for the Finest Text Data at Scale, 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks.
>- A. Radford, et al. Language Models are Unsupervised Multitask Learners, OpenAI, 2019.
>- T. Brown, et al. Language Models are Few-Shot Learners. OpenAI, 2020.
>- A. Kazemnejad, et al. The Impact of Positional Encoding on Length Generalization in Transformers, 37th Conference on Neural Information Processing Systems (NeurIPS 2023).
>- S. Rajbhandari, et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, 2020.
>- J. R. Hermans, et al. Accumulated Gradient Normalization, JMLR: Workshop and Conference Proceedings, 2017.
>- T. Chen, et al. Training Deep Nets with Sublinear Memory Cost. MIT, 2019.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- J. Kaplan, et al. Scaling Laws for Neural Language Models, OpenAI, 2020.
>- J. Hoffman, et al. Training Compute-Optimal Large Language Models, Deep Mind, 2022.
