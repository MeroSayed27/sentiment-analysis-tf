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

## Future Improvements

Planned improvements for this project include:

- Real-time sentiment inference using streaming text inputs
- REST API deployment with FastAPI
- Cloud deployment workflows
- Automated validation testing
- Scalable inference pipelines for production environments

The long-term goal is to evolve this project from a local prediction tool into a deployable real-time NLP service.
