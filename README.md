# Generate a title for a given review - using Seq2Seq Model


## Introduction

**Sequence to Sequence Models** take a sequence of items (words, letters, time series, etc) and output another sequence of items. They are also known as **Encoder-Decoder** models because they use both parts of the Transformer architecture. 

Such models are best suited for tasks revolving around generating new sentences depending on a given input, such as _summarization, translation, or generative question answering_.

## About this Project

This project showcases a **Text Summarizer** which as the name suggests, outputs a summary for a given text input. To take it up a notch, this particular summarizer has been fine tuned specifically to **generate a title for a given review**. 

**Input :** A product review

**Output :** Meaningful title for the review

## Dataset Used

<a href="https://huggingface.co/datasets/amazon_reviews_multi">Amazon Multilingual Reviews Dataset</a>

## Output

Some outputs of the final model are shown below. 

**Note:** Original label shows the original title from the dataset and review is the input for the model.

  ![image](https://github.com/aakanshadalmia/Seq2Seq-Models/assets/35634210/72dc403b-70ab-4c05-aaa9-4304bcd08cf7)

  ![image](https://github.com/aakanshadalmia/Seq2Seq-Models/assets/35634210/b23afda9-557f-44a8-818c-a01410dc5e7c)
  

## Steps to run the model

**1. Install the required modules**

  pip install datasets transformers transformers[sentencepiece] <br>
  pip install --upgrade accelerate <br>
  pip install rouge_score <br>
  pip install nltk <br>
  pip install evaluate

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


  
  



