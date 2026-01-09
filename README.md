# CC-Tuning
The repository for "CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning" (ACL 2025)


<p align="center">
  <img src="Assets/method.png" width="750px" >
</p>


## Environment Setup

```
1. conda create -n CC-Tuning python=3.11
2. conda activate CC-Tuning
3. pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
4. cd ./LLaMA_Factory
5. pip install -e ".[torch,metrics,deepspeed]"
6. pip install jsonlines
```

## Usage

### 1. Training

```
1. cd ./Training
2. llamafactory-cli train train.yaml    # tip: deepspeed does not support 1 gpu
```

### 2. Get Transform Matrix

```
1. python get_vectors.py --model_name {path of the model after CC-Tuning} --save_folder {folder name 1}
2. python get_matrix.py --read_folder {folder name 1} --save_folder {folder name 2}
```


### 3. Inference with Transform Matrix

```
python inference_matrix.py  \
        --dataset XNLI \
        --model {path of the model after CC-Tuning} \
        --matrix_folder llama \
        --save_folder llama
```

We would like to thank [@hiyouga](https://github.com/hiyouga), the training code is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Citation
If you find our work useful, please cite the following paper~
```
@inproceedings{ye-etal-2025-cc,
    title = "{CC}-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning",
    author = "Ye, Yangfan  and
      Feng, Xiaocheng  and
      Yuan, Zekun  and
      Feng, Xiachong  and
      Qin, Libo  and
      Huang, Lei  and
      Ma, Weitao  and
      Huang, Yichong  and
      Zhang, Zhirui  and
      Lu, Yunfei  and
      Yan, Xiaohui  and
      Tang, Duyu  and
      Tu, Dandan  and
      Qin, Bing",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.933/",
    doi = "10.18653/v1/2025.acl-long.933",
    pages = "19036--19051",
    ISBN = "979-8-89176-251-0",
    abstract = "Current large language models (LLMs) often exhibit imbalanced multilingual capabilities due to their English-centric training corpora. To address this, existing fine-tuning approaches operating at the data-level (e.g., through data augmentation or distillation) typically introduce implicit cross-lingual alignment, overlooking the potential for more profound, latent-level cross-lingual interactions. In this work, we propose CC-Tuning, a novel multilingual fine-tuning paradigm that explicitly establishes a cross-lingual connection mechanism at the latent level. During training, CC-Tuning fuses the feed forward activations from both English and non-English inputs, enabling the model to benefit from both linguistic resources. This process is facilitated with a trainable Decision Maker that identifies beneficial activations. Furthermore, during inference, a Transform Matrix is utilized to simulate the cross-lingual connection under monolingual setting through representation transformation. Our experiments on six benchmarks covering 22 languages show that CC-Tuning outperforms vanilla SFT and offers a strong latent-level alternative to data-level augmentation methods. Further analysis also highlights the practicality of CC-Tuning and the potential of latent-level cross-lingual interactions in advancing the multilingual performance of LLMs."
}
```