FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN pip install --no-cache-dir \
    runpod \
    transformers==4.45.0 \
    peft \
    accelerate \
    soundfile \
    librosa \
    huggingface_hub

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
