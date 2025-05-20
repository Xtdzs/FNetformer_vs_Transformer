# FNetformer_vs_Transformer

## Main Work

Based on the paper ***[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)***, I reproduced FNet&Transformer architecture and made an experiment on them. (I reproduced the Transformer Architecture on a very badic level by tensorflow)

## Environment

The experimental environment for this experiment is as follows:

- **Operating System:** Win11
- **Integrated Development Environment (IDE):** PyCharm
- **Python Version:** 3.8.17
- **TensorFlow Version:** 2.6.0
- **CUDA Version:** 11.2
- **cuDNN Version:** 8.1
- **GPU:** NVIDIA GeForce GTX 3060

## Dataset

[Language Translation (English-French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)

### Source

The dataset used in this experiment is sourced from Kaggle and pertains to the task of language translation, specifically translating English text to French. 

### Content(copied from Kaggle)

There are 2 columns. One column has English words/sentences and the other one has French words/sentences

This dataset can be used for language translation tasks.

The unique values are different because the same English word has a different French representation

**Example**: Run(english) = 1. Courezâ€¯! 2. Coursâ€¯!

## Results

I have run 40 epochs with a batch size of 64 for fitting 5000 short sentences with two types of architecture specifically on my local computer.

The summary of Transformer(left) and FNetformer(right) is as below: 

<img src=".\images\BaseCut.png" width="50%"><img src=".\images\FNetCut.png" width="50%">

Here comes the visible comparison on the accuracy and time cost of the two architectures:



<img src=".\images\AccuracyComp.png">

<img src=".\images\TimeCostComp.png">

We could basically confirm that FNetformer is faster than Transformer with a similar accuracy.
