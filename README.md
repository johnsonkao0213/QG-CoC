# QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models


![MultiModal Reasoning](https://img.shields.io/badge/Task-MultiModal_Reasoning-red) 
![MMIU](https://img.shields.io/badge/Dataset-MMIU-blue)  
![MuirBench](https://img.shields.io/badge/Dataset-MuirBench-blue)
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green) 
![Gemini-Flash](https://img.shields.io/badge/Model-Gemini--Flash-green)


[EMNLP'25]
This is the official implementation of `QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models`.

Authors: [Kuei-Chun Kao](https://johnsonkao0213.github.io/), Tzu-Yin Hsu, Yunqi Hong, [Ruochen Wang](https://ruocwang.github.io/), [Cho-Jui Hsieh](https://web.cs.ucla.edu/~chohsieh/)

[[Webpage](https://johnsonkao0213.github.io/QG-CoC/)] [[Paper](https://aclanthology.org/2025.emnlp-main.1445/)]


## Outlines

- [👀 About QG-CoC](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#-about-beyondx)
- [🔮 Evaluations on Benchmark via QG-CoC](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#-evaluations-on-beyondx-via-formulate-and-solve)
  - [Requirements](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#requirements)
  - [Evaluation Pipelines](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#evaluation-pipelines)
- [📈 Evaluation Results](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#-evaluation-results)
- [📜 License](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#-license)
- [✅ Cite](https://github.com/johnsonkao0213/QG-CoC/blob/main/README.md#-cite)


## 👀 About Formulate-and-Solve

**Formulate-and-Solve** is an automated prompting method designed for LLMs to solve math problems with an arbitrary number of unknowns. It incorporates a set of principles to instruct LLMs in generating demonstrations automatically and empowers LLMs to translate problems into equations and subsequently utilize external tools (Sympy) to solve them.

<div align="center">
<img src="docs/static/images/examples/qgcoc_pipeline.png" width="60%">
</div>


## 🔮 Evaluations on Benchmark via QG-CoC

### Installation
Clone the repository and install the requirements.

```bash
git clone https://github.com/johnsonkao0213/QG-CoC.git
cd QG-CoC
```

#### Closed-source models
```bash
pip install -r requirements/base.txt
```
Set your API keys in `.env` file:
```bash
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
export GEMINI_API_KEY=(YOUR GEMINI APIKEY)
```

#### Open-source models

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

### Usage

#### Stpe1: Generate Image Captioning
```bash
python generate_description.py --engine {model_name} --dataset {dataset_name} --domain {domain_name} --task {task_name}
bash run_desc.sh
```

#### Step2: Combine Reasoning
```bash
python inference.py --engine {model_name} --dataset {dataset_name} --domain {domain_name} --task {task_name} --prompting_method {prompting_method} 
bash run.sh (closed-source models)
bash run_open_source.sh (open-source models)
```

#### Step3: Benchmark Evaluation
```bash
python evaluate.py --engine {model_name} --dataset {dataset_name} --domain {domain_name} --task {task_name} --prompting_method {prompting_method}
bash eval.sh
```

## 📈 Evaluation Results

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

<div align="center">
<img src="docs/static/images/mwp_solver.png" width="60%">
</div>

## License

The new contributions to our dataset are distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The copyright of the questions belongs to the original authors, and the source of every original question can be found in the `source` field. Alongside this license, the following conditions apply:

- **Purpose:** The dataset was primarily designed for use as a test set.
- **Commercial Use:** The dataset can be used commercially as a test set, but using it as a training set is prohibited. By accessing or using this dataset, you acknowledge and agree to abide by these terms in conjunction with the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.


## :white_check_mark: Cite

If you find **QG-CoC** useful for your your research and applications, please kindly cite using this BibTeX:
```latex
@inproceedings{kao-etal-2025-qg,
    title = "{QG}-{C}o{C}: Question-Guided Chain-of-Captions for Large Multimodal Models",
    author = "Kao, Kuei-Chun  and
      Tzu-Yin, Hsu  and
      Hong, Yunqi  and
      Wang, Ruochen  and
      Hsieh, Cho-Jui",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1445/",
    pages = "28433--28448",
    ISBN = "979-8-89176-332-6",
    abstract = "Recently, Multimodal Large Language Models (MLLMs) encounter two key issues in multi-image contexts: (1) a lack of fine-grained perception across disparate images, and (2) a diminished capability to effectively reason over and synthesize information from multiple visual inputs. However, while various prompting methods aim to describe visual content, many existing studies focus primarily on single-image settings or specific, constrained scenarios. This leaves a critical gap in understanding and addressing how MLLMs tackle more general and complex multi-image reasoning tasks. Thus, we first extensively investigate how current prompting methods perceive fine-grained visual details and process visual information when dealing with multiple images. Our findings reveal that existing prompting methods fall short in attending to needed clues and seamlessly integrating perception and reasoning. Inspired by the findings, we propose a new zero-shot prompting method, Question-Guided Chain-of-Captions (QG-CoC), a generalized prompting approach that effectively handles problems with an arbitrary number of images. We evaluate our method on various open-source and closed-source MLLMs for multi-image and single-image benchmarks. Experimental results indicate that QG-CoC demonstrates competitive performance across tasks and exhibits robust improvements in the challenging scenarios where existing prompting methods fail."
}
```

