---
runme:
  id: 01HQP1RTRHBTQ46VZV21Y8P666
  version: v3
---

# CPSC542 Assignment 1

## Student Information
- **Student Name:** Devyn Miller
- **Student ID:** 2409539

## Collaboration
- **In-class collaborators:** Received help from Caitlyn, the TA.

## Resources
- **Resources used:** 
    - "Convolutional Neural Networks: A Brief History of their Evolution"
    - "A Comprehensive Survey on Transfer Learning"
    - "Meta-Learning in Neural Networks: A Survey"
    - [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
    - [Why One-Hot Encode Data in Machine Learning?](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
    - [How to Develop a CNN from Scratch for CIFAR-10 Photo Classification](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)
    - [CIFAR-10 with CNN for Beginner](https://www.kaggle.com/code/roblexnana/cifar10-with-cnn-for-beginer/notebook)
    - Copilot
    - Perplexity

## Data Source
- **Data source:** CIFAR-10 dataset from Keras: `from keras.datasets import cifar10`

## Code Repository
- **GitHub Repository:** [CPSC542-cifar-Assignment1](https://github.com/devyn-miller/assignment-1-cpsc-542.git)

# Additional notes below — please read.

1. All of the plots and figures for this project can be found in the Jupyter notebook entitled `main.ipynb`. The report only contains the graphs for the loss function and accuracy.

2. When reviewing the `main.ipynb` file, please note that the exploratory data analysis (EDA) is separated from the `main()` function for the reasons mentioned above. The EDA function is provided as a standalone section within the notebook and can be executed independently to gain insights into the dataset.

3. I have included a folder called `not-for-submission` which contains supplementary Jupyter notebooks that were created during the development and experimentation process for the assignment. These notebooks are not part of the main submission but provide insights into the iterative development and exploration of different approaches. Each notebook represents a different version or variation of the main assignment, exploring different techniques, parameters, or design choices.

Contents:

- `CPSC542_cifar_Assignment1SeparableConv2D.ipynb`: Notebook exploring the use of separable convolutional layers.
- `CPSC542_cifar_Assignment1_grayscale.ipynb`: Notebook experimenting with grayscale conversion of images.
- `CPSC542_cifar_Assignment1batch_128.ipynb`: Notebook investigating batch size effects with a batch size of 128.
- `CPSC542_cifar_Assignment1downsample_image_res.ipynb`: Notebook exploring downsampled image resolutions.
- `CPSC542_cifar_Assignment1extra_convblock.ipynb`: Notebook with additional convolutional blocks for comparison.
- `CPSC542_cifar_Assignment1only_one_conv_block.ipynb`: Notebook focusing on using only one convolutional block.
- `CPSC542_cifar_Assignment1v2.ipynb`: Revised version of the main assignment with improvements or changes.
- `CPSC542_cifar_Assignment1v2_(2)only_two_conv_blocks_finished_running.ipynb`: Notebook with only two convolutional blocks, completed and ready for review.

These notebooks are provided for reference purposes and can be consulted to understand the development process, experiments conducted, and decisions made during the assignment. They may contain incomplete or experimental code and should not be considered as final submissions.