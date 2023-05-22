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

