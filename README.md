# GAZK : GPU accelerated ZK-SNARKs

This is a class project for EECS-4750 Heterogenous Computing taught by Prof. Zoran Kostic. In this project we accelerate the performance of the prover in PLONK based ZK-SNARK systems. 

ZK_SNARKs are a cryptographic primitive that allow one party who runs a computation to ~prove~ to another party that it ran that computation via a short proof, or witness string. The other party then can run a short computation, called a verifier that validates this string to determine whether or not the proof is accurate. While the verifier is computationally cheap, the prover is quite resource intensive, and proportional to the actual computation being run (but manifold more). ZK-SNARKS allow a party with limited resources (say a mobile phone) to offload their computation to a stronger party (like a cloud computing platform), while maintaining guarantees that the computation was run accurately by the other party. This protocol allows for trustless and verifiable computation.

We identified parts of the PLONK proving protocol that take the longest. Our goal for this project is to introduce experiments that show how we can utilise GPUs to parallelise this computation and introduce faster provers that make this technology more practical.

## How to install

After cloning the repository, run the following commands:
```
make virtualenv
source venv/bin/activate
make all
```
This will install the GAZK package. Note that the experiments using CUDA kernels require CUDA installed. Please follow the instructions here [INSERT LINK] to setup CUDA, and be sure to have a GPU available.

## Codebase
The codebase has the following structure:
[INSERT FILE TREE]

The experiments folder will let you produce timing analysis. Files flagged with GPU need CUDA enabled to run.


