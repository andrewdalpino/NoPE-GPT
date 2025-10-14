# NoPE GPT

NoPE GPT is a generative pretrained Transformer-style (GPT) language model with no positional embeddings (NoPE). Built using [PyTorch](https://pytorch.org/) and trained on HuggingFace's [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), and [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) datasets, NoPE GPT can answer questions, summarize documents, use tools, and more.

## Features

- **No positional embeddings (NoPE)**: NoPE GPT aims to be a more parsimonious model by completely removing positional embeddings from the architecture allowing the context length to vary without complex model surgery. Despite having no positional embeddings, NoPE GPT performs better at context length generalization than the best relative embeddings (ALiBi, RoPE, T5) offering good performance even when operating within 2X the trained context window.

- **Fast and memory-efficient**: NoPE GPT employs a number of training and inference-time optimizations such as Group Query Attention (GQA), Flash Attention, KV-caching, activation checkpointing, and fully-sharded data parallel (FSDP) pretraining. As such, you can train and infer on relatively modest hardware.

- **Fully Open-source**: Unlike closed-source LLMs, NoPE GPT provides both the model weights *and* the source code to train, fine-tune, export, and generate text from the model using your own hardware.

## Pretrained Models

| Name | Context Length | Vocab. Size | Embedding Dim. | Query Heads | Key/Value Heads | Layers |
|---|---|---|---|---|---|---|
| [NoPE-GPT-400M-Chat](https://huggingface.co/andrewdalpino/NoPE-GPT-400M-Chat) | 8192 | 50,261 | 1280 | 20 | 5 | 20 |
| [NoPE-GPT-400M-Base](https://huggingface.co/andrewdalpino/NoPE-GPT-400M-Base) | 8192 | 50,257 | 1280 | 20 | 5  | 20 |

## Installation

The code required to run inference comes as a Python package that you can install with your favorite package manager such as [pip](https://pypi.org/project/pip/).

```sh
pip install nope-gpt
```

## Pretrained Examples

This first example we'll show how to load a pretrained base model from HuggingFace Hub and then use it to generate text. First, make sure the `nope-gpt` package is installed into your project. Once the package is installed you can load pretrained weights from HuggingFace Hub like in the example below.

```python
from nope_gpt.model import NoPEGPT
from nope_gpt.tokenization import BaseTokenizer

model_name = "andrewdalpino/NoPE-GPT-400M-Base"

model = NoPEGPT.from_pretrained(model_name)

tokenizer = BaseTokenizer.from_pretrained(model_name)
```

Then, to generate text, provide a prompt, tokenize it, and iterate through the `generate()` method until the model outputs a stop token.

```python
import torch

prompt = input("Enter a prompt: ")

prompt = tokenizer.tokenize(prompt)

prompt = torch.tensor(prompt, dtype=torch.int64)

for token, probability in model.generate(prompt):
    if token.item() in tokenizer.stop_tokens:
        break

    out = tokenizer.decode_single_token(token)

    print(out, end="", flush=True)
```

Generating text from the base model is the simplest way to get started with model inference, however, it is not the most useful when it comes to being able to chat with and guide the model output. In this example we'll load one of the pretrained chat models from HuggingFace Hub and then chat with it. In addition, we'll make use of short-term memory so the model can remember the chat history.

First, load a pretrained chat model from HuggingFace Hub like in the example below.

```python
from nope_gpt.model import NoPEGPT
from nope_gpt.tokenization import ChatTokenizer

model_name = "andrewdalpino/NoPE-GPT-400M-Chat"

model = NoPEGPT.from_pretrained(model_name)

tokenizer = ChatTokenizer.from_pretrained(model_name)
```

Then, we'll define a partial function that will generate tokens with a set of default parameters such as `max_tokens`, `context_length`, and `temperature`.

```python
from functools import partial

generate = partial(
    model.generate,
    max_tokens=2000,
    context_length=8192,
    temperature=0.7,
    top_k=500
    top_p=0.9,
    repeat_penalty=0.1,
    repeat_window=50,
)
```

Next, we'll instantiate a `BufferWindowMemory` object to handle the chat history and craft a system message that will guide generation. Note that messages are inputted as dicts with `role` and `content` keys. For a system message use the `system` role.

```python
from nope_gpt.memory import BufferWindowMemory

memory = BufferWindowMemory(4)

system_message = {
    "role": "system",
    "content": "You are a friendly AI assistant.",
}
```

Finally, prompt the user for input, adds the system message and chat history to the context, tokenizes the messages, and then generates the `assistant` response.

```python
import torch

while True:
    prompt = input("Enter a prompt: ")

    user_message = {
        "role": "user",
        "content": prompt,
    }

    memory.add_message(user_message)

    messages = [system_message] + memory.get_history()

    tokens = tokenizer.tokenize_prompt(messages)

    prompt = torch.tensor(tokens, dtype=torch.int64, device=args.device)

    response = ""

    for token, probability in generate(prompt):
        if token.item() in tokenizer.stop_tokens:
            break

        out = tokenizer.decode_single_token(token)

        print(out, end="", flush=True)

        response += out

    print("\n")

    assistant_message = {
        "role": "assistant",
        "content": response,
    }

    memory.add_message(assistant_message)
```

You're done! For more advanced usages take a look at the `generate.py` and `chat.py` scripts located in the code repository.

## Training and Fine-tuning

In addition to the inference code, we also provide training and fine-tuning code so you can build your own NoPE GPT models. Before getting started, take a look at the `model_sizing.ipynb` IPython notebook in the project repo for a guide to sizing your model based on the amount of memory and compute you have available.

### Clone the project repo

We'll need the code from the project repository to train and/or fine-tune the model.

```
git clone https://github.com/andrewdalpino/NoPE-GPT
```

### Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

### Pretraining

Pretraining focuses on building a foundation of language and general knowledge to use as a base for future supervised fine-tuning. The training objective is to predict the next token in a sample of text. It is a self-supervised form of training because the model learns from masked inputs of unsupervised data. For the pretraining corpus we use the Fineweb dataset which consists of 15T high-quality tokens gathered from the worldwide web. In addition, the dataset has been split into 3 subsets (10BT, 100BT, and 350BT versions) for training smaller models.

```
python pretrain.py
```

**Note** that it will take a while to download and pre-process the dataset the first time that the training script is run.

To customize the default architecture you can adjust the `embedding_dimensions`, attention heads, `num_hidden_layers`, and `feed_forward_ratio` arguments of the pretraining script. 

```
python pretrain.py --embedding_dimensions=4096 --num_q_heads=64 --num_kv_heads=16 --num_hidden_layers=48 --feed_forward_ratio=4
```

You can also adjust the `batch_size`, `learning_rate`, and `gradient_accumulation_steps` to suite your training setup.

```
python pretrain.py --batch_size=32 --learning_rate=0.01 --gradient_accumulation_steps=128
```

If you are planning a long training run, it is recommended to set a random seed. This will ensure that any random state is preserved if the process gets interrupted.

```
python pretrain.py --seed=42
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
| --batch_size | 2 | int | The number of samples of size `tokens_per_sample` to pass through the network at a time. |
| --gradient_accumulation_steps | 128 | int | The number of batches to pass through the network before updating the model weights. |
| --tokens_per_sample | 4096 | int | The number of tokens to pack into a single training sequence. This is sometimes called the block size or context length. |
| --max_steps | 10000 | int | The maximum number of steps to take for pretraining. |
| --learning_rate | 1e-2 | float | The learning rate of the Adafactor optimizer. |
| --anneal_learning_rate | False | bool | Should we linearly decay the learning rate to zero as the step approaches `max_steps`? |
| --low_memory_optimizer | False | bool | Should the optimizer reduce its memory consumption in exchange for a slightly slower runtime? |
| --max_gradient_norm | 10.0 | float | Clip gradients above this threshold norm before stepping. |
| --embedding_dimensions | 1024 | int | The dimensionality of the token embeddings. |
| --num_q_heads | 16 | int | The number of query heads within every attention layer. |
| --num_kv_heads | 4 | int | The number of key and value heads within every attention layer. |
| --num_hidden_layers | 16 | int | The number of attention/MLP blocks within the body of the network. |
| --feed_forward_ratio | 4 | (1, 2, 4) | The ratio of hidden neurons to embedding dimensions in the MLP layers of the network. |
| --dropout | 0.0 | float | The proportion of signals to send to zero during training as regularization. |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will drastically reduce memory utilization during training at the cost of recomputing the forward pass. |
| --ddp_sharding_level | 2 | int | The level of sharding to use for DDP training. Options are 2 or 3 for partial and full sharding respectively, or 0 for no sharding. |
| --eval_interval | 100 | int | Evaluate the model after this many epochs on the testing set. |
| --num_eval_samples | 2048 | int | The number of hold-out samples to use for validation during training. |
| --checkpoint_interval | 100 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --resume_scheduler | False | bool | Should we resume the learning rate schedule from the last checkpoint? |
| --run_dir_path | "./runs" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cpu" | str | The device to run the training on ex `cuda`, `cuda:0`, `mps`, `cpu`. |
| --seed | None | int | The seed for the random number generator. |

### Fine-tuning

Instruction-tuning is a supervised training technique focused on developing specialized objectives such as chatting, text summarization, chain-of-thought, and prompt rewriting. We use the SmolTalk and UltraFeedback datasets by HuggingFace as fine-tuning corpora because they include a broad range of training objectives such as conversation, instruction following, summarization, and human preference alignment.

```
python fine-tune.py
```

To pick which dataset subsets to train on you can specify them in a comma-separated list like in the example below.

```
python fine-tune.py --dataset_subsets=smol-magpie-ultra,smol-summarize,ultra-feedback
```

You can also adjust the `batch_size`, `learning_rate`, and `gradient_accumulation_steps` just like we did with pre-training.

```
python fine-tune.py --batch_size=32 --learning_rate=0.01 --gradient_accumulation_steps=32
```

To adjust the number of trainable LoRA parameters as well as the strength of the LoRA and Dropout signals you can change the `--rank` and `--alpha` arguments respectively.

```
python fine-tune.py --rank=4 --alpha=2.0
```

To quantize the base model weights during fine-tuning (QLoRA) you can specify the `quantize_base_weights` argument and to adjust the quant group size set the `quant_group_size` argument like in the example below.

```
python fine-tune.py --quantize_base_weights --quant_group_size=128
```

In memory constrained environments, you can enable activation checkpointing to trade off compute for memory efficiency by recomputing the activations of each decoder block during backpropagation.

```
python fine-tune.py --activation_checkpointing
```

### Fine-tuning Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --base_checkpoint_path | None | string | The path to the base model checkpoint on disk. |
| --dataset_subset | "all" | str | A comma-separated list of subsets of the dataset to train on. Options are `all`, `apigen-80k`, `everyday-conversations`, `explore-instruct-rewriting`, `longalign`, `metamathqa-50k`, `numina-cot-100k`, `openhermes-100k`, `self-oss-instruct`, `smol-constraints`, `smol-magpie-ultra`, `smol-rewrite`, `smol-summarize`, `systemchats-30k`, and `ultra-feedback`. |
| --max_tokens_per_sample | 4096 | int | The maximum number of tokens to pack into a single training sequence. |
| --filter_long_samples | False | bool | Should we filter out samples that are longer than the max_tokens_per_sample? |
| --num_dataset_processes | 8 | int | The number of processes to use for processing the dataset. |
| --batch_size | 1 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 128 | int | The number of batches to pass through the network before updating the weights. |
| --num_epochs | 2 | int | The number of epochs to train for. |
| --learning_rate | 1e-2 | float | The learning rate of the Adafactor optimizer. |
| --low_memory_optimizer | False | bool | Should the optimizer reduce its memory consumption in exchange for a slightly slower runtime? |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --rank | 8 | int | The rank of the LoRA decomposition matrices. |
| --alpha | 1.0 | float | The strength of the LoRA signal. |
| --freeze_token_embeddings | False | bool | Should we freeze the weights of the token embeddings? |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will reduce drastically memory utilization during training at the cost of needing to recompute the forward pass. |
| --eval_interval | 1 | int | Evaluate the model after this many epochs on the testing set. |
| --num_eval_samples | 2048 | int | The number of hold-out samples to use for validation during training. |
| --checkpoint_interval | 1 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the model checkpoint. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cpu" | str | The device to run the training on ex `cuda`, `cuda:0`, `mps`, `cpu`. |
| --seed | None | int | The seed for the random number generator. |

### Training Dashboard

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to capture and display pretraining events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```
tensorboard --logdir=./runs
```

Then navigate to the dashboard using your favorite web browser.

## References:
>- G. Penedo, et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale, 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks.
>- L. B. Allal, et al. SmolLM2 - with great data, comes great performance, 2024.
>- A. Radford, et al. Language Models are Unsupervised Multitask Learners, OpenAI, 2019.
>- T. Brown, et al. Language Models are Few-Shot Learners. OpenAI, 2020.
>- A. Kazemnejad, et al. The Impact of Positional Encoding on Length Generalization in Transformers, 37th Conference on Neural Information Processing Systems (NeurIPS 2023).
>- S. Rajbhandari, et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, 2020.
>- J. R. Hermans, et al. Accumulated Gradient Normalization, JMLR: Workshop and Conference Proceedings, 2017.
>- T. Chen, et al. Training Deep Nets with Sublinear Memory Cost. MIT, 2019.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- J. Kaplan, et al. Scaling Laws for Neural Language Models, OpenAI, 2020.
>- J. Hoffman, et al. Training Compute-Optimal Large Language Models, Deep Mind, 2022.
>- J. Ainslie, et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Google Research, 2023.
>- S. Bergsma, et al. Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs, ICLR, 2025.
>- R. Zhao, et al. Deconstructing What Makes a Good Optimizer for Autoregressive Language Models, ICLR, 2025.
