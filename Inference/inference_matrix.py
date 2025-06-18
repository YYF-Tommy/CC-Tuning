import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import jsonlines
from tqdm.auto import tqdm
import argparse
import time
import numpy as np
from collections import Counter
import time
from load_dataset import get_data
import os

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


def generate_text_loop(prompt, max_new_tokens=512):

    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # aux (en)
        if "Llama-3" in model_name:
            print("llama3_occurrence")
            occurrence_indices = find_occurrence(input_ids, "llama3")
        elif "Qwen" in model_name:
            print("qwen_occurrence")
            occurrence_indices = find_occurrence(input_ids, "qwen")
        elif "Mistral" in model_name:
            print("mistral_occurrence")
            occurrence_indices = find_occurrence(input_ids, "mistral")
        
        inputs = {
            "cached_vectors": None,
            "occurrence_indices_aux": occurrence_indices,
            "input_ids": input_ids,
        }
        model(**inputs)

        cached_vectors_aux = []  # en
        for item in model.model.layers:
            cached_vectors_aux.append(item.mlp_output)

        print(cached_vectors_aux[0].shape)
        cached_vectors_aux = torch.stack(cached_vectors_aux)

        cached_vectors_aux = cached_vectors_aux @ torch.from_numpy(matrix).to("cuda")

        cached_vectors_aux = [cached_vectors_aux[i, :] for i in range(cached_vectors_aux.shape[0])]
 
        new_tokens = None
        generated_new_text = ""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        past_key_values = None
        
        for i in range(max_new_tokens):
            if i > 0:
                cached_vectors_aux = None
            outputs = model(
                cached_vectors=cached_vectors_aux,
                occurrence_indices_aux=None,
                input_ids=input_ids.to(device),
                past_key_values=past_key_values, 
                use_cache=True 
            )
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_id = logits[:, -1, :].argmax(dim=-1)

            if next_token_id == config.eos_token_id:
                break
            
            if new_tokens != None:
                new_tokens = torch.cat((new_tokens, next_token_id))
            else:
                new_tokens = next_token_id

            generated_new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            input_ids = next_token_id.unsqueeze(0) 

    return generated_new_text


if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--matrix_folder", type=str)
    parser.add_argument("--save_folder", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset
    save_folder = args.save_folder
    matrix_folder = args.matrix_folder

    device = "cuda"

    model_name = args.model
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = model.config

    print(model)

    langs_all = {
                 "XNLI": ["en", "ar", "el", "hi", "ru", "sw", "th", "tr", "ur", "zh"],
                 "XStoryCloze": ["ar", "en", "es", "eu", "hi", "id", "ru", "sw", "te", "zh"],
                 "MMMLU": ["en", "ar", "bn", "es", "hi", "id", "ko", "pt", "sw", "yo"],
                 "XQuAD": ["ar", "de", "el", "en", "hi", "ru", "th", "tr", "vi", "zh"],
                 "MKQA": ["en", "ar", "de", "ja", "ko", "pt", "ru", "tr", "vi", "zh"],
                 "XLSum": ["en", "ar", "fr", "hi", "id", "ru", "sw", "tr", "ur", "vi"]
                 }
    
    langs = langs_all[dataset_name]

    path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/Inference/output/{dataset_name}/{save_folder}"
    matrix_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/saved_matrix/{matrix_folder}.npy"
    matrix = np.load(matrix_path)

    start_time = time.time() 

    for lang in langs:
        prompts_lang, prompts_en = get_data(dataset_name, lang)
        outputs = []
        count = 0
        for prompt_lang in tqdm(prompts_lang):
            prompt_lang = template(model_name, prompt_lang)

            output = generate_text_loop(prompt_lang, max_new_tokens=40)
            outputs.append(output)

        os.makedirs(path, exist_ok=True)

        with jsonlines.open(f"{path}/{lang}.json", "w") as f:
            for line in outputs:
                f.write(line)

        # break
    end_time = time.time()  

    execution_time = end_time - start_time 
    print(dataset_name)
    print(f"Execution time: {execution_time:.4f} seconds")