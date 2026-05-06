# Movie/TV Review Classifier

This project trains a two-stage classifier for three labels:

- `0` = not a movie/TV review
- `1` = positive review
- `2` = negative review

## Run directly

Put `train.csv` and `test.csv` in the project root, then run:

```bash
python train_classifier.py
```

The script will:

- print local validation results
- train the final model
- write `submission.csv`
- save model files in `artifacts/`

## Docker

Build the container:

```bash
docker build -t review-classifier .
```

Run it:

```bash
docker run --rm -v "$(pwd):/app" review-classifier
```
