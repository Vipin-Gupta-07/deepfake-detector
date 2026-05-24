---
title: DeepScan - Deepfake Detector API
emoji: 🔍
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DeepScan — Deepfake Image Detection API

A FastAPI backend for detecting deepfake facial images using a fine-tuned **ViT-B/16** Vision Transformer with **MTCNN** face preprocessing.

## Endpoints

- `GET /` — Health check
- `POST /predict` — Upload an image for Real/Fake classification

## Architecture

- **MTCNN** (facenet-pytorch) for face detection & cropping
- **ViT-B/16** pretrained on ImageNet-21k (via TIMM) for binary classification
- **97.1% accuracy** / AUC 0.991 on Kaggle 140k Real & Fake Faces dataset

## Frontend

The web UI is hosted on GitHub Pages:
**https://vipin-gupta-07.github.io/deepfake-detector/**
