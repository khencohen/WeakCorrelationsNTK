# Weak Correlations as the Underlying Principle for Linearization of Gradient-Based Learning Systems

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Deep learning models, such as wide neural networks, can be conceptualized as nonlinear dynamical physical systems characterized by a multitude of interacting degrees of freedom. Such systems, in the limit of an infinite number of degrees of freedom, tend to exhibit simplified dynamics. This paper delves into gradient descent-based learning algorithms that display a linear structure in their parameter dynamics, reminiscent of the neural tangent kernel. We establish that this apparent linearity arises due to weak correlations between the first and higher-order derivatives of the hypothesis function with respect to the parameters taken around their initial values. This insight suggests that these weak correlations could be the underlying cause behind the observed linearization in such systems. As a case in point, we showcase this weak correlations structure within neural networks in the large width limit. Utilizing this relationship between linearity and weak correlations, we derive a bound on the deviation from linearity during the training trajectory of stochastic gradient descent. To facilitate our proof, we introduce a novel method to characterize the asymptotic behavior of random tensors. We empirically verify our findings and present a comparison between the linearization of the system and the observed correlations.


In this repository we provide the code for tests in the paper:



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)


## Installation
git clone https://github.com/khencohen/WeakCorrelationsNTK.git

cd WeakCorrelationsNTK

pip install -r requirements.txt



## Usage
python main.py



## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yourPaperID,
  title={Your Paper Title},
  author={Your Name and Co-Authors},
  journal={Journal Name or Conference},
  year={Year},
  doi={YourDOI}
}
