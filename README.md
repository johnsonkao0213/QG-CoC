# QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models


![MultiModal Reasoning](https://img.shields.io/badge/Task-MultiModal_Reasoning-red) 
![MMIU](https://img.shields.io/badge/Dataset-MMIU-blue)  
![MuirBench](https://img.shields.io/badge/Dataset-MuirBench-blue)
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green) 
![Gemini-Flash](https://img.shields.io/badge/Model-Gemini--Flash-green)


[EMNLP'25]
This is the official implementation of `QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models`.

Authors: [Kuei-Chun Kao](https://johnsonkao0213.github.io/), Tzu-Yin Hsu, Yunqi Hong, [Ruochen Wang](https://ruocwang.github.io/), [Cho-Jui Hsieh](https://web.cs.ucla.edu/~chohsieh/)

<!-- Abstract -->



## Installation
Clone the repository and install the requirements.

```bash
git clone {repo_url}
cd {repo_name}
```

### Closed-source models
```bash
pip install -r requirements/base.txt
```
Set your API keys in `.env` file:
```bash
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
export GEMINI_API_KEY=(YOUR GEMINI APIKEY)
```

### Open-source models

Install the corresponding environment for each model.
```bash
pip install -r requirements/{model_name}.txt
```

You may want to use conda to create a new environment for each model.

```bash
conda create -n {model_name} python=3.10
conda activate {model_name}
pip install -r requirements/{model_name}.txt
```

## Usage

### Generate Image Captioning
```bash
python generate_description.py --engine {model_name} --dataset {dataset_name} --domain {domain_name} --task {task_name}
```

### Inference
```bash
python inference.py --engine {model_name} --dataset {dataset_name} --domain {domain_name} --task {task_name} --prompting_method {prompting_method} 
```

### Evaluation
```bash
python evaluate.py --engine {model_name} --dataset {dataset_name} --domain {domain_name} --task {task_name} --prompting_method {prompting_method}
```

## Experiments

### Different Models

#### Closed-source models
* GPT-4o
* Gemini-1.5-Flash

#### Open-source models
* LLaVA-OneVision-7B
* Mantis-Idefics2-7B
* Qwen-2.5-VL-7B

### Different Datasets

#### Multi-Image
* MMIU
* MUIRBench

#### Single-Image
* MMBench
* ScienceQA
* MMMU


### [Related Paper](https://hackmd.io/H84oJdNwTe6vvZq-R6y1Ow?both)

