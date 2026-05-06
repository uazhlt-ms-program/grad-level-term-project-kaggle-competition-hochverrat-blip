# Movie/TV Review Classifier

This project trains a two-stage text classifier for three labels:

- `0`: not a movie/TV review
- `1`: positive review
- `2`: negative review

The model uses only the provided text:

- Stage 1: review vs not review
- Stage 2: positive vs negative for texts predicted to be reviews

## Local run

Create a virtual environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_classifier.py
```

Running the script will:

- print local validation metrics
- train on the full training data
- write `submission.csv`
- save fitted models in `artifacts/`

## Docker run

Build the image:

```bash
docker build -t review-classifier .
```

Run the classifier:

```bash
docker run --rm review-classifier
```

After the container finishes, `submission.csv` and `artifacts/` will exist inside the container filesystem. If you want them on the host machine, mount the project directory:

```bash
docker run --rm -v "$(pwd):/app" review-classifier
```

This project was developed with Python 3.12 for the direct-run path. The Docker image also uses Python 3.12 so the runtime is controlled and reproducible.

## GitHub Actions

The repository includes a GitHub Actions workflow that:

- creates a Python 3.12 environment
- runs `train_classifier.py`
- builds the Docker image
- runs the Docker container

This provides an automated Linux verification path in addition to the local instructions above.
