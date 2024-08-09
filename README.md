# Scalable Synthetic Data Generation (SDG)

## Introduction

Framework for scalable [synthetic data generation (SDG)](https://en.wikipedia.org/wiki/Synthetic_data).

## Getting Started

### Setup

We recommend using a Python virtual environment with Python 3.9+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html):

```
python3 -m venv ssdg_venv
source ssdg_venv/bin/activate
pip install .
```

**Note:** If you have used [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or another tool for Python version management, then use the virtual environment with that tool instead. Otherwise, you may have issues with packages installed but modules from that package not found as they are linked to you Python version management tool and not `venv`.

SDG uses Large Language Models (LLMs) to generate synthetic data. Scalable SDG therefore requires access to LLMs to inference or call the model. The following LLM inference APIs are supported:

- [IBM Generative AI (GenAI)](https://ibm.github.io/ibm-generative-ai/v3.0.0/index.html)
- [OpenAI](https://github.com/openai/openai-python)
- [vLLM](https://github.com/vllm-project/vllm)

Scalable SDG uses a `.env` file to specify the configuration for the IBM GenAI and OpenAI APIs. The `.env` file needs to be availabe from where the generate command is run from. There is a template `env` file [here](https://github.com/foundation-model-stack/fms-sdg/blob/main/.env.example).

The subsections that follow explain how to setup for the different APIs.

#### IBM Generative AI (GenAI)

When using the IBM GenAI API, you need to:

1. Add configuration to `env` file as follows:

```yaml
GENAI_KEY=<genai key goes here>
GENAI_API=<genai api goes here>
```

2. Install GenAI dependencies as follows:

```command
pip install -e ".[genai]"
```

#### OpenAI

When using the OpenAI platform, you need to:

1. Add configuration to `env` file as follows:

```yaml
OPENAI_API_KEY=<openai api key goes here>
```

2. Install OpenAI dependencies as follows:

```command
pip install -e ".[openai]"
```

#### vLLM

When using the vLLM batched inference, you need to:

1. Install vLLM dependencies as follows:

```command
pip install -e ".[vllm]"
```

#### Data Prep Kit

When using the postprocessing pipeline supported by Data Prep Kit (DPK), you need to install:

```command
pip install -e ".[dpk]"
```

**Note:** vLLM [requires Linux OS and CUDA](https://docs.vllm.ai/en/latest/getting_started/installation.html#requirements).

### Testing out the Framework

To get started with this example, make sure you have followed the [Setup](#setup) instructions, [configured IBM GenAI](#ibm-generative-ai-genai), and/or [configured vLLM](#vLLM)

In this example, we will use the preloaded data files as the seed data to to generate the synthetic data.

#### Testing with GenAI

The default data builder is set to run with the GenAI api unless overridden. We thus only need to run the following command (run from the root of the repository) to execute data generation with GenAI:

```command
python -m fms_dgt.__main__ --data-paths ./data/generation/logical_reasoning/causal/qna.yaml
```

Alternatively, you can also use the CLI

```command
fms_dgt --data-paths ./data/generation/logical_reasoning/causal/qna.yaml
```

If you want to pass multiple data paths, you can do so

```command
fms_dgt --data-paths ./data/generation/logical_reasoning/causal ./data/writing/freeform
```

#### Testing with vLLM

For convenience, we have provided an additional configuration file that can be modified to test out using a local model with vLLM. First, open [the config file](./configs/demo.yaml) and update the model field `model_id_or_path` to substitute the `<local-path-to-model>` variable with the path of a model that has been downloaded locally.

```command
python -m fms_dgt.__main__ --data-paths ./data/generation/logical_reasoning/causal/qna.yaml --include-config-path ./configs/demo.yaml
```

**Note:** vLLM [requires Linux OS and CUDA](https://docs.vllm.ai/en/latest/getting_started/installation.html#requirements).

#### Examine Outputs

The generated data will be output to the following directory: `output/causal/data->logical_reasoning->causal/generated_instructions.json`

This example uses the `SimpleInstructDataBuilder` as defined in `./fms_dgt/databuilders/generation/simple/`. For more information on data builders and other components of Scalable SDG, take a look at the [SDG Design](./docs/sdg_design.md) doc.

## Contributing

Check out our [contributing](./CONTRIBUTING.md) guide to learn how to contribute.

## References

This repository is based on the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) which uses an MIT license.

```citation
@misc{eval-harness,
    author = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
    title = {A framework for few-shot language model evaluation},
    month = 12,
    year = 2023,
    publisher = {Zenodo},
    version = {v0.4.0},
    doi = {10.5281/zenodo.10256836},
    url = {https://zenodo.org/records/10256836}
}
```
