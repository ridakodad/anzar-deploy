FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel

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
