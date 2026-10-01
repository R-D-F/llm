import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace this with a local folder containing config.json and *.safetensors,
# or leave it unchanged to download the model from Hugging Face.
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-4B")


def main():
    print(f"Loading {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "8GiB", "cpu": "16GiB"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Model metadata
    print("\n=== CONFIGURATION ===")
    print(model.config.to_json_string())

    # Weight names, shapes, types, and locations
    print("\n=== PARAMETERS ===")
    for name, parameter in model.named_parameters():
        print(
            f"{name:70} "
            f"shape={str(tuple(parameter.shape)):24} "
            f"dtype={parameter.dtype} device={parameter.device}"
        )

    # Locate and inspect one weight tensor.
    name, weight = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("layers.0.mlp.down_proj.weight")
    )

    print(f"\n=== INSPECTING {name} ===")
    print(weight[:3, :5].float())

    # Set TWEAK_WEIGHT=1 to make this in-memory modification.
    if os.getenv("TWEAK_WEIGHT") == "1":
        with torch.no_grad():
            before = weight[0, 0].item()
            weight[0, 0] = 0
            after = weight[0, 0].item()

        print(f"Changed {name}[0, 0]: {before} -> {after}")

    print("\nEnter an empty prompt to quit.")

    while True:
        prompt = input("\nYou: ").strip()
        if not prompt:
            break

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0, inputs.input_ids.shape[1] :]
        print("Assistant:", tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
