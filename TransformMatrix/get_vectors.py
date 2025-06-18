import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import jsonlines
from tqdm.auto import tqdm
import argparse
import numpy as np
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LLaMA_Factory.train_utils.tag_locate import find_occurrence


def template(model_name, content):
    if "Llama-3" in model_name:
        print("Apply Llama3 template !")
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "Qwen" in model_name:
        print("Apply Qwen template !")
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    elif "Mistral" in model_name:
        print("Apply Mistral template !")
        return f"<s>[INST] {content} [/INST]"


def get_data():
    prompts_en = []
    prompts_lang = []
    with open(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/Training/data/aya_train_mini_parallel.json") as f:
        data = json.load(f)
    for item in data:
        prompts_lang.append(item["instruction"])
        prompts_en.append(item["instruction_en"])

    random.seed(666)
    select_ids = random.sample(range(len(prompts_lang)), 10)  # sample_num = 1000

    prompts_lang = [prompts_lang[id] for id in select_ids]
    prompts_en = [prompts_en[id] for id in select_ids]

    return prompts_lang, prompts_en


def generate_text_loop(prompt, prompt_aux, max_new_tokens=512):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # vanilla
        if "Llama-3" in model_name:
            print("llama3_occurrence")
            occurrence_indices = find_occurrence(input_ids, "llama3")
        elif "Qwen" in model_name:
            print("qwen_occurrence")
            occurrence_indices = find_occurrence(input_ids, "qwen2")
        elif "Mistral" in model_name:
            print("mistral_occurrence")
            occurrence_indices = find_occurrence(input_ids, "mistral")
        
        inputs = {
            "cached_vectors": None,
            "occurrence_indices_aux": occurrence_indices,
            "input_ids": input_ids,
        }
        model(**inputs)

        count_t = len(os.listdir(f"{save_path}/nonEn"))
        saved_vectors = []
        for item in model.model.layers:
            saved_vectors.append(item.mlp_output[0].clone().cpu().numpy())
        saved_vectors = np.stack(saved_vectors)
        np.save(f"{save_path}/nonEn/{count_t}.npy", saved_vectors)


    with torch.no_grad():
        input_ids_aux = tokenizer(prompt_aux, return_tensors="pt").input_ids.to(device)

        # aux
        if "Llama-3" in model_name:
            print("llama3_occurrence")
            occurrence_indices_aux = find_occurrence(input_ids_aux, "llama3")
        elif "Qwen" in model_name:
            print("qwen_occurrence")
            occurrence_indices_aux = find_occurrence(input_ids_aux, "qwen2")
        elif "Mistral" in model_name:
            print("mistral_occurrence")
            occurrence_indices_aux = find_occurrence(input_ids_aux, "mistral")
        
        inputs_aux = {
            "cached_vectors": None,
            "occurrence_indices_aux": occurrence_indices_aux,
            "input_ids": input_ids_aux,
        }
        model(**inputs_aux)

        count_t = len(os.listdir(f"{save_path}/En"))
        saved_vectors_aux = []
        for item in model.model.layers:
            saved_vectors_aux.append(item.mlp_output[0].clone().cpu().numpy())
        saved_vectors_aux = np.stack(saved_vectors_aux)
        np.save(f"{save_path}/En/{count_t}.npy", saved_vectors_aux)
        


if __name__ == "__main__":
    """
    Usage: python get_vectors.py --model_name {path of the model after CC-Tuning} --save_folder {folder name}
    """
    print(torch.cuda.is_available())

    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B")
    parser.add_argument("--save_folder", type=str, default="Llama-3.1-8B")
    args = parser.parse_args()

    model_name = args.model_name
    print("Model Name: ", model_name)

    if "Llama" in model_name:
        from LLaMA_Factory.src.llamafactory.model.llamawrapper_mlp import LlamaForCausalLM
        load_class = LlamaForCausalLM
    elif "Qwen" in model_name:
        from LLaMA_Factory.src.llamafactory.model.qwen2wrapper_mlp import Qwen2ForCausalLM
        load_class = Qwen2ForCausalLM
    elif "Mistral" in model_name:
        from LLaMA_Factory.src.llamafactory.model.mistralwrapper_mlp import MistralForCausalLM
        load_class = MistralForCausalLM
    
    model = load_class.from_pretrained(model_name).to(device)

    save_folder = args.save_folder


    save_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{save_folder}"
    os.makedirs(f"{save_path}/nonEn", exist_ok=True)
    os.makedirs(f"{save_path}/En", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = model.config

    print(model)

    prompts_lang, prompts_en = get_data()
    outputs = []

    for prompt_lang, prompt_en in tqdm(zip(prompts_lang, prompts_en)):
        prompt_lang = template(model_name, prompt_lang)
        prompt_en = template(model_name, prompt_en)
        generate_text_loop(prompt_lang, prompt_en, max_new_tokens=1)