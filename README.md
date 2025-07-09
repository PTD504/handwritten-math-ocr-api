# Handwritten Math OCR API

This repository provides a complete solution for converting images of handwritten mathematical equations into LaTeX format. It includes a high-performance REST API for inference and a full suite of scripts for training your own models.

The core of the project is a deep learning model built with PyTorch, featuring a Swin Transformer encoder and a Transformer decoder, designed for high accuracy in formula recognition. The API is built with FastAPI and is fully containerized with Docker for easy deployment and scalability.

## Model Architecture

The model uses an encoder-decoder architecture optimized for image-to-sequence tasks:

*   **Encoder:** A **Swin Transformer (Tiny)**, pre-trained on ImageNet, is adapted to process single-channel (grayscale) formula images. It extracts robust hierarchical features from the input image.
*   **Decoder:** A standard **Transformer Decoder** generates the corresponding LaTeX sequence token by token, attending to the features extracted by the encoder.

## API Endpoints

The service exposes several RESTful endpoints for prediction, monitoring, and diagnostics.

| Endpoint           | Method | Description                                                                         |
|--------------------|--------|-------------------------------------------------------------------------------------|
| `/predict`         | `POST` | Upload an image file (`.png`, `.jpg`, etc.) to receive its LaTeX representation.    |
| `/predict/batch`   | `POST` | Send a list of base64-encoded image strings for batch processing.                   |
| `/status`          | `GET`  | Get the operational status of the API, including model and vocabulary load status.  |
| `/health`          | `GET`  | Perform a detailed health check of the service and its components.                  |
| `/model/info`      | `GET`  | Retrieve the configuration and parameters of the loaded model.                      |

## Getting Started: Running the API

You can quickly get the API running locally using Docker.

### Prerequisites

*   Git
*   Docker and Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/ptd504/handwritten-math-ocr-api.git
cd handwritten-math-ocr-api
```

### 2. Place Model Files

The API requires a trained model and a vocabulary file to function.

1.  Train your own model by following the [Training](#training) instructions below.
2.  Once training is complete, a `best_model.pth` and `vocab.json` will be saved in the `checkpoints/` directory.
3.  Copy these files into the `app/trained-model/` directory. Rename `best_model.pth` to `model.pth`.

Your `app/trained-model/` directory should look like this:

```
app/trained-model/
├── model.pth
└── vocab.json
```

### 3. Run with Docker Compose

With the model files in place, start the service using Docker Compose.

```bash
docker-compose --file app/docker-compose.yml up --build
```

The API will be available at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

### 4. Make a Prediction

Use `curl` or any HTTP client to send an image to the `/predict` endpoint.

```bash
curl -X POST -F "file=@/path/to/your/formula.png" http://localhost:8000/predict
```

The response will be a JSON object containing the predicted LaTeX formula:

```json
{
  "formula": "\\frac { d } { d x } e ^ { x } = e ^ { x }",
  "confidence": 0.9876,
  "processing_time": 0.543,
  "timestamp": "2023-10-27 10:30:00"
}
```

## Training

If you wish to train the model on your own dataset, you can use the provided training scripts.

### 1. Setup Environment

Install the required Python packages from the root `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 2. Prepare the Data

Organize your dataset according to the structure described in `data/README.md`. This typically involves creating directories for images (`train_formulas`, `validate_formulas`) and corresponding CSV files (`train_labels.csv`, `validate_labels.csv`) with image filenames and their LaTeX labels.

### 3. Build the Vocabulary

Generate a `vocab.json` file from your training labels. This file maps each unique token in your dataset to an index.

```bash
python src/build_vocab.py
```

This will create `vocab.json` inside the `checkpoints/` directory.

### 4. Start Training

Run the training script. You can choose between a standard training loop or one integrated with MLflow for experiment tracking.

**Standard Training:**

```bash
python src/train.py
```

**Training with MLflow:**

```bash
python src/train_mlflow.py
```

The script will train the model, validate it at the end of each epoch, and save checkpoints. The model with the best validation performance will be saved as `best_model.pth` in the `checkpoints/` directory.

### 5. Use the Trained Model

After training is complete, copy the `best_model.pth` (renamed to `model.pth`) and `vocab.json` from the `checkpoints/` directory to the `app/trained-model/` directory to use them with the API.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.