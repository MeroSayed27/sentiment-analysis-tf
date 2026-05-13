# Sentiment Analysis with Python & TensorFlow

This project predicts whether a text review is **Positive** or **Negative**
using a trained deep learning model.

## How to run
```bash
py -3.10 predict.py

## Docker Support

This project now includes a Docker configuration to support reproducible deployment across different environments.

### Build Docker image
```bash
docker build -t sentiment-analysis .
```

### Run container
```bash
docker run -it sentiment-analysis
```

This structure makes it easier to scale the application consistently across local systems, cloud infrastructure, and edge environments.
