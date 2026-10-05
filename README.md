<h1 align="center">CARE</h1>

<p align="center"><strong>Context-Adaptive Rubric Evolution</strong></p>

<p align="center">
  <a href="https://asa13opp.github.io/CARE-project-page/"><img src="https://img.shields.io/badge/Project-Page-2ea44f?logo=githubpages&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/collections/0tishuang/care"><img src="https://img.shields.io/badge/Models-HuggingFace-yellow?style=plastic&amp;logo=Hugging%20Face" alt="Datasets &amp; Models on Hugging Face"></a>
  <a href="https://arxiv.org/abs/2609.34249"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=plastic&amp;logo=arxiv" alt="Paper on arXiv"></a>
</p>

<p align="center">Turn-adaptive reward specifications for empathetic reinforcement learning.</p>

<p align="center"><a href="docs/">Project page source</a> · <a href="docs/README.md">GitHub Pages deployment</a></p>

CARE generates a rubric for each dialogue turn, adapting the weights and criteria for **cognitive, affective, and proactive empathy**. The rubric generator is trained with supervised fine-tuning and preference-based GRPO, then used to reward policy responses during online GRPO.

## Pipeline

| Stage | Purpose | Code |
|---|---|---|
| Rubric SFT | Learn turn-level rubric structure and criteria | `rubric_gen/scripts/`, `rubric_gen/configs/sft/` |
| Rubric preference GRPO | Align rubrics with human response preferences | `rubric_gen/rl/`, `rubric_gen/configs/rl/` |
| Online policy GRPO | Score responses with context-adaptive rubrics | `online_rl/src/`, `online_rl/configs/` |

**Implementation scope:** this checkout includes the CARE training stages and an RLVER-style online runner. MICA integration, upstream data annotation/collection, and benchmark evaluation assets/scripts are not included.

## Quickstart

Requires Linux, NVIDIA CUDA, and compatible PyTorch, Transformers, DeepSpeed, veRL, and vLLM dependencies. Set `CONDA_SH` and `CONDA_ENV` for your environment. Configure API endpoints and credentials in the relevant local `*.env` files. Model variables accept Hugging Face IDs or local paths.

### 1. Train the rubric generator

Uses `rubric_gen/data/sft/{train,val}.jsonl` (2 GPUs by default).

```bash
cd rubric_gen
CONDA_SH=/path/to/conda.sh \
CONDA_ENV=verl \
MODEL_PATH=/path/to/base-model \
bash configs/sft/train.sh
```

### 2. Align rubrics with preferences

Uses `rubric_gen/data/rl/{train,val}.parquet` (4 GPUs by default). Configure `rubric_gen/configs/rl/judge.env` first.

```bash
cd rubric_gen
CONDA_SH=/path/to/conda.sh \
CONDA_ENV=verl \
MODEL_PATH=/path/to/sft-checkpoint \
bash configs/rl/run_grpo.sh
```

### 3. Optimize the dialogue policy

Uses `online_rl/data/{train,test}_profile.jsonl`. Configure `online_rl/configs/rubric.env` and `online_rl/configs/judge.env`.

```bash
CONDA_SH=/path/to/conda.sh \
CONDA_ENV=verl \
POLICY_MODEL=/path/to/policy-model \
bash online_rl/configs/run_online_rubric_rl.sh
```

Set `SMOKE=1` for a one-step infrastructure check. It uses mock rubric generation, judging, and simulation; its output is not an experimental result.

## Citation

```bibtex
@misc{huang2026evolving,
  title         = {Evolving Support Priorities in Empathetic Reinforcement Learning},
  author        = {Huang, Pengyu and Han, Zhiyuan and Tong, Wenwen and Guo, Hewei and Chen, Jiangnan and Chen, Sirui and Lu, Lewei and Zhu, Beier and Yang, Xun},
  year          = {2026},
  eprint        = {2609.34249},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

Questions or feedback: [otishuang@163.com](mailto:otishuang@163.com)
