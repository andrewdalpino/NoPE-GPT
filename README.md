# GPT

A Generative Pre-trained Transformer (GPT) model trained on Openwebtext to generate text when prompted using PyTorch's Distributed Data Parallel protocol for training over multiple CUDA-enabled GPU clusters. The default implementation is similar to GPT-2 by OpenAI and can be easily customized and scaled up to meet your needs and compute budget. In addition, you may incorporate your own training data alongside Openwebtext for additional pre-training samples or for fine-tuning for a specific task after pre-training.

## Download the Repository
Clone the project locally using git:

```
git clone https://github.com/andrewdalpino/GPT
```

## Requirements

- [Python](https://www.python.org/) 3.10 or later
- A CUDA-enabled GPU with 12G of VRAM or more

## Recommended

- A CUDA-enabled GPU cluster with 40G of VRAM or more

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
python train.py
```

If you have a larger system you can increase the training load by increasing the `batch_size` at runtime.

```
python train.py --batch_size=8
```

To distribute the training workload over a cluster or multiple cluster nodes, use PyTorch's [torchrun](https://pytorch.org/docs/stable/elastic/run.html) extension to launch a distributed data parallel session. When training in data-parallel mode it's important that the `gradient_accumulation_steps` divides evenly into the world size for maximum performance. For example, if we have an 8 GPU cluster, we could perform 32 gradient accumulation steps in exactly 4 passes over the network.

```
torchrun train.py --batch_size=16 --gradient_accumulation_steps=32
```

After training, you can generate text from the model by running the `generate.py` script from the commandline with a prompt.

```
python generate.py --prompt="When something is important enough, you should do it even if the odds are not in your favor."
```

### Training Parameters

| Argument | Default | Type | Description |
|---|---|---|---|
| --batch_size | 4 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 32 | int | The number of batches to pass through the network before updating the weights. |
| --max_samples_per_epoch | 4096 | int | The maximum number of training samples to pass through the network every epoch. |
| --learning_rate | 5e-4 | float | The global step size taken after every gradient accumulation step. |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold before stepping. |
| --num_epochs | 1000 | int | The number of epochs to train for. |
| --eval_epochs | 10 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_epochs | 20 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./out/ckpt.pt" | string | The path to the checkpoint file on disk. |
| --dataset_path | "./dataset" | string | The path to the dataset files on disk. |
| --num_dataset_processes | 4 | int | The number of processes (CPUs) to use to process the dataset. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator . |

### Generation Parameters

| Argument | Default | Type | Description |
|---|---|---|---|
| --prompt | "\n" | string | The text prompt that the model should complete. |
| --max_tokens | 300 | int | The maximum number of tokens that the model should generate per sample. |
| --temperature | 1.0 | float | The amount of regularization applies to the candidate token probabilities. |
| --top_k | 200 | int | Only sample from this many candidate tokens with the highest probabilities. |
| --checkpoint_path | "./out/ckpt.pt" | string | The path to the checkpoint file on disk. |
| --device | "cuda" | string | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator . |

## References:
>- A. Vaswani, et al. Attention Is All You Need. 31st Conference on Neural Information Processing Systems, 2017.
>- A. Radford, et al. Language Models are Unsupervised Multitask Learners.
>- O. Press, et. al. Using the Output Embedding to Improve Language Models.

## License
The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).