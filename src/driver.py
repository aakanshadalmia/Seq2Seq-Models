import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)

from argument_parser import parse_args
from baseline import evaluate_baseline
from data_processing import load_data, tokenize_data

nltk.download("punkt")


def train(args):
    english_dataset = load_data("amazon_reviews_multi", "en")
    model_card = args.model_card
    model = AutoModelForSeq2SeqLM.from_pretrained(model_card)
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    tokenized_data = english_dataset.map(
        lambda example: tokenize_data(example, tokenizer),
        batched=True,
        remove_columns=list(english_dataset["train"].features),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [
            "\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
        ]
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: round(value * 100, 4) for key, value in result.items()}
        return result

    rouge_score = evaluate.load("rouge")

    batch_size = args.batch_size
    logging_steps = len(tokenized_data["train"]) // batch_size
    model_name = model_card.split("/")[-1]

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_dir}/{model_name}-finetuned-amazon-en-es",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    performance = trainer.evaluate()
    score = evaluate_baseline(english_dataset["validation"], evaluate.load("rouge"))
    print(f"\nRouge scores for lead-3 baseline {score}")
    print(f"\nRouge scores for trained model {performance}")

    best_model_checkpoint = trainer.state.best_model_checkpoint
    print(f"Best model checkpoint: {best_model_checkpoint}")
    return best_model_checkpoint


def predict(best_model_checkpoint):
    english_dataset = load_data("amazon_reviews_multi", "en")
    text_summarizer = pipeline("summarization", best_model_checkpoint)
    input_texts = english_dataset["test"]["review_body"]
    labels = english_dataset["test"]["review_title"]
    for input_text, label in zip(input_texts, labels):
        if len(input_text.split(" ")) >= 100:
            print(
                f"Label: {label} \nOutput: {text_summarizer(input_text)[0]['summary_text']}\n"
            )


if __name__ == "__main__":
    args = parse_args()
    print("Arguments", args)
    train(args)
