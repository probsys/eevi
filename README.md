# Estimators of Entropy and Information via Inference in Probabilistic Models

This repository contains experiment code for

_Estimators of Entropy and Information via Inference in Probabilistic Models__.
Feras A. Saad, Marco Cusumano Towner, Vikash K. Mansinghka.
Proceedings of The 25th International Conference on Artificial Intelligence
and Statistics, PMLR 151:5604-5621, 2022.
https://proceedings.mlr.press/v151/saad22a.html

## Getting started

1. Install julia v1.6.2 from

    https://github.com/JuliaLang/julia/releases/tag/v1.6.2

2. Set current directory to the Julia project using

    export JULIA_PROJECT=.

3. Instantiate the package dependencies using

    julia -e 'using Pkg; Pkg.instantiate()'

    The main dependency is the Gen.jl package,

## Running experiments

Please navigate to [./examples](./examples) and follow the README.

These experiments show how to estimate the entropy of random variables or
the (conditional) mutual information between groups of random variables in
a probabilistic program written in [Gen.jl](https://www.gen.dev/). The two
applications in [./examples](./examples) directory are based on Gen
probabilistic programs that encode models for blood glucose monitoring and
the HEPAR expert system for liver disease.

A further reference of the program analysis implementation in Gen can be
found in Section 8.6 of the following dissertation:

_Scalable Structure Learning, Inference, and Analysis with Probabilistic
Programs_. Feras A. K. Saad. PhD Thesis, Massachusetts Institute of
Technology, 2022. Pages 178–182.
https://dspace.mit.edu/handle/1721.1/147226
