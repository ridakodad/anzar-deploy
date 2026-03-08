import runpod
import base64
import tempfile
import os
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel

print("Loading base model...")
BASE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
LORA_MODEL = "DrIAmed/Anzar-2.0/qwen2-audio-darija-merged"
HF_TOKEN = os.environ.get("HF_TOKEN")

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL,
    token=HF_TOKEN
)
model.eval()
print("Model ready!")


def handler(job):
    job_input = job["input"]
    audio_b64 = job_input.get("audio")
    prompt = job_input.get("prompt", "Please transcribe this audio in Darija and translate it to English.")

    if not audio_b64:
        return {"error": "No audio provided. Send base64 encoded WAV."}

    try:
        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": tmp_path},
                    {"type": "text",  "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        audios = [tmp_path]
        inputs = processor(
            text=text,
            audios=audios,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        os.unlink(tmp_path)
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
