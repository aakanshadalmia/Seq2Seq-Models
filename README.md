# Summary Generation for reviews

## Introduction

**Sequence to Sequence Models** take a sequence of items (words, letters, time series, etc) and output another sequence of items. They are also known as **Encoder-Decoder** models because they use both parts of the Transformer architecture. 

Such models are best suited for tasks revolving around generating new sentences depending on a given input, such as _summarization, translation, or generative question answering_.

## About this Project

This project showcases a **Text Summarizer** which as the name suggests, outputs a summary for a given text input. To take it up a notch, this particular summarizer has been fine tuned specifically to **generate a title for a given review**. 

**Input :** Review for a product

**Output :** Meaningful short summary for the review

## Dataset Used

<a href="https://huggingface.co/datasets/amazon_reviews_multi">Amazon Multilingual Reviews Dataset</a>

## Model Description

### Overview
A multilingual Text-to-Text Transfer Transformer (mT5) model 
has been used in this project. 

### About the Model
mT5 is basically a multilingual variant of T5 that has been pre-trained on a Common Crawl-based dataset covering 101 languages. The model architecture and training procedure that
we use for mT5 closely follows that of T5.

T5 is a pre-trained language model whose primary distinction is its use of a unified “text-to-text” format for all text-based NLP problems. This approach is natural for generative tasks where the task format requires the model to generate text conditioned on some input. 

Given the sequence-to-sequence structure of this task format,
T5 uses a basic encoder-decoder Transformer architecture as proposed by [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)

## Steps to run the final model

**1. Install the required modules**

  To get started, clone this repository and run the below command to make sure all required modules are installed.
  
  ```
  pip install -r requirements.txt
  ```
  
**2. Run `driver.py`**
  
  Commonly modified arguments have been configured in `argument_parser.py` to be passed as command line arguments.
  - `model_card` Model to be used, _default = "google/mt5-small"_
  - `batch_size` Size of batch, _default = 32_
  - `weight_decay` Weight decay, _default = 0.01_
  - `learning_rate ` Learning rate, _default = 5.6e-5_
  - `save_total_limit` Number of checkpoints to save, _default = 3_
  - `num_train_epochs` Number of training epochs, _default = 3_
  - `output_dir` Output Directory, _default = "."_

  **Note**: All above mentioned arguments are optional, to be used as and when required.
  
  Example:
  
  ```
  python driver.py --model_card "google/mt5-base" --learning_rate 2e-5 --batch_size 16 --num_train_epochs 4
  ```

## Output

Some outputs of the final model are shown below. 

**Note:** Original label shows the original title from the dataset and review is the input for the model.

  ![image](https://github.com/aakanshadalmia/Seq2Seq-Models/assets/35634210/72dc403b-70ab-4c05-aaa9-4304bcd08cf7)

  ![image](https://github.com/aakanshadalmia/Seq2Seq-Models/assets/35634210/b23afda9-557f-44a8-818c-a01410dc5e7c)
  

  
  



