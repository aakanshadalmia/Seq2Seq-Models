from datasets import load_dataset


def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )


def load_data(dataset_name, language):
    english_dataset = load_dataset(dataset_name, language)
    english_dataset = english_dataset.filter(filter_books)
    return english_dataset


def tokenize_data(example, tokenizer, max_length=256, max_target_length=30):
    model_inputs = tokenizer(
        example["review_body"], max_length=max_length, truncation=True
    )
    label_inputs = tokenizer(
        example["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = label_inputs["input_ids"]
    return model_inputs
