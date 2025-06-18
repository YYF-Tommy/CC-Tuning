import jsonlines
import os

data_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/data"

def get_data(dataset_name, lang):
    if dataset_name == "XNLI":
        return get_xnli(lang)
    elif dataset_name == "XStoryCloze":
        return get_xstorycloze(lang)
    elif dataset_name == "MMMLU":
        return get_mmmlu(lang)
    elif dataset_name == "XQuAD":
        return get_xquad(lang)
    elif dataset_name == "MKQA":
        return get_mkqa(lang)
    elif dataset_name == "XLSum":
        return get_xlsum(lang)
    else:
        return None


def get_xnli(lang):
    prompts_en = []
    prompts_lang = []
    with jsonlines.open(f"{data_path}/XNLI/split_sample/en.json") as f:
        for line in f:
            prompts_en.append(line["instruction"])
    with jsonlines.open(f"{data_path}/XNLI/split_sample/{lang}.json") as f:
        for line in f:
            prompts_lang.append(line["instruction"])
    return prompts_lang, prompts_en


def get_xquad(lang):
    prompts_en = []
    prompts_lang = []
    with jsonlines.open(f"{data_path}/XQuAD/split/en.json") as f:
        for line in f:
            prompts_en.append(line["instruction"])
    with jsonlines.open(f"{data_path}/XQuAD/split/{lang}.json") as f:
        for line in f:
            prompts_lang.append(line["instruction"])
    return prompts_lang, prompts_en


def get_mmmlu(lang):
    prompts_en = []
    prompts_lang = []
    with jsonlines.open(f"{data_path}/MMMLU/split_sample/en.json") as f:
        for line in f:
            prompts_en.append(line["instruction"])
    with jsonlines.open(f"{data_path}/MMMLU/split_sample/{lang}.json") as f:
        for line in f:
            prompts_lang.append(line["instruction"])
    return prompts_lang, prompts_en

def get_xstorycloze(lang):
    prompts_en = []
    prompts_lang = []
    with jsonlines.open(f"{data_path}/XStoryCloze/split/en.json") as f:
        for line in f:
            prompts_en.append(line["instruction"])
    with jsonlines.open(f"{data_path}/XStoryCloze/split/{lang}.json") as f:
        for line in f:
            prompts_lang.append(line["instruction"])
    return prompts_lang, prompts_en

def get_mkqa(lang):
    prompts_en = []
    prompts_lang = []
    with jsonlines.open(f"{data_path}/MKQA/split_sample/en.json") as f:
        for line in f:
            prompts_en.append(line["instruction"])
    with jsonlines.open(f"{data_path}/MKQA/split_sample/{lang}.json") as f:
        for line in f:
            prompts_lang.append(line["instruction"])
    return prompts_lang, prompts_en

def get_xlsum(lang):
    prompts_en = []
    prompts_lang = []
    with jsonlines.open(f"{data_path}/XLSum/split_sample_en/{lang}.json") as f:
        for line in f:
            prompts_en.append(line["instruction"])
    with jsonlines.open(f"{data_path}/XLSum/split_sample/{lang}.json") as f:
        for line in f:
            prompts_lang.append(line["instruction"])
    return prompts_lang, prompts_en