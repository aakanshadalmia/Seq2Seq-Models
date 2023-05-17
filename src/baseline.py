from nltk.tokenize import sent_tokenize


def _three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_baseline(dataset, metric):
    summaries = [_three_sentence_summary(text) for text in dataset["review_body"]]
    score = metric.compute(predictions=summaries, references=dataset["review_title"])
    score = {key: round(value * 100, 4) for key, value in score.items()}
    return score
