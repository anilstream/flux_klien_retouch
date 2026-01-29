FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES 1
ARG HF_TOKEN

# set working directory
WORKDIR /app

# install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    wget \
    gegl \
    unzip \
    libgl1 \
    libglx-mesa0 \
    git-lfs \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN git clone https://github.com/comfyanonymous/ComfyUI
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install -r ComfyUI/requirements.txt
RUN pip3 install fastapi[standard]

# install huggingface
RUN pip install huggingface_hub
RUN hf auth login --token $HF_TOKEN

# download models
RUN wget https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors -O ComfyUI/models/text_encoders/qwen_3_8b_fp8mixed.safetensors
RUN wget https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors -O ComfyUI/models/vae/flux2-vae.safetensors
RUN hf download black-forest-labs/FLUX.2-klein-9b-fp8 flux-2-klein-9b-fp8.safetensors --local-dir ComfyUI/models/diffusion_models

WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch

# copy source files
WORKDIR /app
COPY . .

# run fastapi app
CMD python3 retouch_api.py