<!--   <img src="logo.png" width="200" height="200"> -->

# Autocomp: LLM-driven Code Optimization for Tensor Accelerators

[![arXiv](https://img.shields.io/badge/arXiv-2505.18574-b31b1b.svg)](https://arxiv.org/abs/2505.18574)
[![Blog Post](https://img.shields.io/badge/Blog-github.io-blue)](https://charleshong3.github.io/blog/autocomp.html)

Welcome to the code repository of **Autocomp**. Check out our introductory [blog post](https://charleshong3.github.io/blog/autocomp.html)!

**Paper**: [**Autocomp: LLM-Driven Code Optimization for Tensor Accelerators**](https://arxiv.org/abs/2505.18574)

**Authors**: [Charles Hong](https://charleshong3.github.io/), [Sahil Bhatia](https://x.com/sahilb17), [Alvin Cheung](https://people.eecs.berkeley.edu/~akcheung/), and [Yakun Sophia Shao](https://people.eecs.berkeley.edu/~ysshao/) (UC Berkeley)

Note that this repository is still under construction.

## Setup

Install all necessary dependencies: ``pip install -e .``

Chipyard/FireSim setup instructions coming soon.

## Repository Structure

## Usage

`autocomp/search/search.py` can be used to run Autocomp for GEMM and convolution benchmarks, but TinyMPC kernels (stored under the name `admm-multifunction`) require manual changes to the code.

## Citations
```
@misc{hong2025autocomp,
      title={Autocomp: LLM-Driven Code Optimization for Tensor Accelerators}, 
      author={Charles Hong and Sahil Bhatia and Alvin Cheung and Yakun Sophia Shao},
      year={2025},
      eprint={2505.18574},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18574}, 
}
```
