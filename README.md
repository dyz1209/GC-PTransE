# Readme

# GC-PTransE

This repository contains the code implementation for the experiments described in the paper "GC-PTransE: Multi-Step Attack Inference

Method Based on Graph Convolutional Neural Network and Translation Embedding".

## Datasets

We have collected threat intelligence data related to 7 categories of APT organizations. The current data consists of unclassified data, as well as classified data for the 7 categories of APTs that will be used for subsequent inference. Detailed data files can be found in the data directory.（Due to security concerns, we are only able to disclose a portion of the cybersecurity data.）

## Model

### Recommended Configuration

- Python: 3.8
- Torch：2.1.0
- Dgl：2.2.1

### Additional Packages Required

Install packages by yourself if they are not already installed Recommended dependencies

## Training

Regarding the training execution, the `main.py` script is executed, and the command run in the terminal is `python main.py` (under the root directory of this project).
