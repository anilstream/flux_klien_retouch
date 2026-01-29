# Flux Klien Retouch

SAM3 API provides promptable image segmentation using flexible input modalities such as point clicks, bounding boxes, and natural language text prompts. In addition, the API includes a mask refinement endpoint for generating high-quality refined alpha masks.

## Install

Build the flux klien retouch fastapi app with docker and push it to dockerhub.

* **Build**: docker build --build-arg HF_TOKEN=$HF_TOKEN -t anilsathyan/flux-klien-retouch:latest .
* **Login**: docker login
* **Push**: docker push anilsathyan/flux-klien-retouch:latest
* **Run**: docker run -d -p 5007:5007 --runtime=nvidia --gpus all anilsathyan/flux-klien-retouch:latest