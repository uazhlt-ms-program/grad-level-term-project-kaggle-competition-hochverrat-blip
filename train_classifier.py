import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


LABEL_NOT_REVIEW = 0
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 2

STAGE1_C = 4.0
STAGE2_C = 16.0
STAGE1_THRESHOLD = 0.49
STAGE2_THRESHOLD = 0.51

HTML_BREAK_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w']+\b|[.!?;,]")
ALPHA_TOKEN_RE = re.compile(r"[A-Za-z']+")
FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", flags=re.IGNORECASE)
REVIEW_CUE_RE = re.compile(
    r"\b(recommend|recommended|worth|waste|boring|amazing|awful|great|terrible|love|hate|liked|disliked)\b",
    flags=re.IGNORECASE,
)
SUMMARY_CUE_RE = re.compile(
    r"\b(stars?|starring|directed|author|pages?|published|edition|season|episode)\b",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CONTRAST_SPLIT_RE = re.compile(r"\b(?:but|however|though|although|yet)\b", flags=re.IGNORECASE)
NEGATION_CUES = {
    "no",
    "not",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "cannot",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "shouldn't",
    "wouldn't",
    "couldn't",
    "hasn't",
    "haven't",
    "hadn't",
}
NEGATION_ENDERS = {".", "!", "?", ";", ","}


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = html.unescape(text)
    text = HTML_BREAK_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def emphasize_boundaries(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return normalized
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if not sentences:
        return normalized
    if len(sentences) == 1:
        return f"{sentences[0]} {normalized} {sentences[0]}"
    return f"{sentences[0]} {normalized} {sentences[-1]}"


def emphasize_boundaries_and_contrast(text: str) -> str:
    emphasized = emphasize_boundaries(text)
    if not emphasized:
        return emphasized
    contrast_chunks = CONTRAST_SPLIT_RE.split(emphasized)
    if len(contrast_chunks) <= 1:
        return emphasized
    trailing_chunks = [chunk.strip() for chunk in contrast_chunks[1:] if chunk.strip()]
    if not trailing_chunks:
        return emphasized
    contrast_tail = " ".join(trailing_chunks)
    return f"{emphasized} {contrast_tail}"


def extract_stage1_structural_features(texts):
    rows = []
    for raw_text in texts:
        original = "" if pd.isna(raw_text) else str(raw_text)
        normalized = normalize_text(original)
        alpha_tokens = ALPHA_TOKEN_RE.findall(normalized)
        word_count = len(alpha_tokens)
        char_count = len(normalized)
        sentence_punct_count = sum(normalized.count(mark) for mark in ".!?")
        exclam_count = normalized.count("!")
        question_count = normalized.count("?")
        html_break_count = len(HTML_BREAK_RE.findall(original))
        first_person_count = len(FIRST_PERSON_RE.findall(normalized))
        review_cue_count = len(REVIEW_CUE_RE.findall(normalized))
        summary_cue_count = len(SUMMARY_CUE_RE.findall(normalized))
        uppercase_chars = sum(1 for ch in original if ch.isupper())
        alpha_chars = sum(1 for ch in original if ch.isalpha())
        uppercase_ratio = uppercase_chars / alpha_chars if alpha_chars else 0.0
        avg_word_len = (sum(len(tok) for tok in alpha_tokens) / word_count) if word_count else 0.0
        rows.append(
            [
                np.log1p(word_count),
                np.log1p(char_count),
                np.log1p(sentence_punct_count),
                np.log1p(exclam_count),
                np.log1p(question_count),
                np.log1p(html_break_count),
                first_person_count / max(word_count, 1),
                review_cue_count / max(word_count, 1),
                summary_cue_count / max(word_count, 1),
                uppercase_ratio,
                avg_word_len,
            ]
        )
    return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))


def default_word_tokenize(text: str):
    return [token for token in WORD_RE.findall(text.lower()) if token not in NEGATION_ENDERS]


def negation_word_tokenize(text: str):
    tokens = WORD_RE.findall(text.lower())
    output = []
    negate = False
    negation_window = 0
    for token in tokens:
        if token in NEGATION_ENDERS:
            negate = False
            negation_window = 0
            continue
        if token in NEGATION_CUES:
            output.append(token)
            negate = True
            negation_window = 3
            continue
        if negate and negation_window > 0:
            output.append(f"NEG_{token}")
            negation_window -= 1
        else:
            output.append(token)
        if negation_window == 0:
            negate = False
    return output


def build_pipeline(c_value, tokenizer, preprocessor, structural_extractor=None):
    transformers = [
        (
            "word_tfidf",
            TfidfVectorizer(
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                token_pattern=None,
                strip_accents="unicode",
                lowercase=True,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                preprocessor=preprocessor,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                sublinear_tf=True,
            ),
        ),
    ]
    if structural_extractor is not None:
        transformers.append(
            (
                "structural_features",
                FunctionTransformer(structural_extractor, validate=False),
            )
        )
    return Pipeline(
        steps=[
            ("features", FeatureUnion(transformer_list=transformers)),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_models(train_df: pd.DataFrame):
    stage1_target = (train_df["LABEL"] != LABEL_NOT_REVIEW).astype(int)
    stage1_model = build_pipeline(
        STAGE1_C,
        default_word_tokenize,
        normalize_text,
        extract_stage1_structural_features,
    )
    stage1_model.fit(train_df["TEXT"], stage1_target)

    review_df = train_df[train_df["LABEL"] != LABEL_NOT_REVIEW].copy()
    stage2_model = build_pipeline(
        STAGE2_C,
        negation_word_tokenize,
        emphasize_boundaries_and_contrast,
    )
    stage2_model.fit(review_df["TEXT"], review_df["LABEL"])
    return stage1_model, stage2_model


def predict(stage1_model: Pipeline, stage2_model: Pipeline, texts: pd.Series):
    review_probs = stage1_model.predict_proba(texts)[:, 1]
    positive_index = list(stage2_model.classes_).index(LABEL_POSITIVE)
    positive_probs = stage2_model.predict_proba(texts)[:, positive_index]

    preds = []
    for review_prob, positive_prob in zip(review_probs, positive_probs):
        if review_prob < STAGE1_THRESHOLD:
            preds.append(LABEL_NOT_REVIEW)
        elif positive_prob >= STAGE2_THRESHOLD:
            preds.append(LABEL_POSITIVE)
        else:
            preds.append(LABEL_NEGATIVE)
    return preds


def print_validation_results(stage1_model: Pipeline, stage2_model: Pipeline, eval_df: pd.DataFrame):
    stage1_true = (eval_df["LABEL"] != LABEL_NOT_REVIEW).astype(int)
    stage1_probs = stage1_model.predict_proba(eval_df["TEXT"])[:, 1]
    stage1_pred = (stage1_probs >= STAGE1_THRESHOLD).astype(int)
    preds = predict(stage1_model, stage2_model, eval_df["TEXT"])

    print(f"Stage 1 F1: {f1_score(stage1_true, stage1_pred):.4f}")
    print(f"Accuracy: {accuracy_score(eval_df['LABEL'], preds):.4f}")
    print(f"Macro F1: {f1_score(eval_df['LABEL'], preds, average='macro'):.4f}")
    print(classification_report(eval_df["LABEL"], preds, digits=4))


def main():
    train_df = pd.read_csv("train.csv")[["ID", "TEXT", "LABEL"]].copy()
    test_df = pd.read_csv("test.csv")[["ID", "TEXT"]].copy()

    train_df["TEXT"] = train_df["TEXT"].fillna("")
    test_df["TEXT"] = test_df["TEXT"].fillna("")
    train_df["LABEL"] = train_df["LABEL"].astype(int)

    fit_df, eval_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=42,
        stratify=train_df["LABEL"],
    )

    stage1_model, stage2_model = train_models(fit_df)
    print_validation_results(stage1_model, stage2_model, eval_df)

    stage1_model, stage2_model = train_models(train_df)
    preds = predict(stage1_model, stage2_model, test_df["TEXT"])

    submission = pd.DataFrame({"ID": test_df["ID"], "LABEL": preds})
    submission.to_csv("submission.csv", index=False)

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stage1_model, artifact_dir / "stage1_review_vs_not_review.joblib")
    joblib.dump(stage2_model, artifact_dir / "stage2_positive_vs_negative.joblib")


if __name__ == "__main__":
    main()
