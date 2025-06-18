# CC-Tuning
The repository for "CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning" (ACL 2025)


<p align="center">
  <img src="Assets/method.png" width="750px" >
</p>


## Environment Setup

```python
1. conda create -n CC-Tuning python=3.11
2. conda activate CC-Tuning
3. pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
4. cd ./LLaMA_Factory
5. pip install -e ".[torch,metrics,deepspeed]"
6. pip install jsonlines
```

## Usage

### 1. Training

```python
1. cd ./Training
2. llamafactory-cli train train.yaml    # tip: deepspeed does not support 1 gpu
```

### 2. Get Transform Matrix

```python
1. python get_vectors.py --model_name {path of the model after CC-Tuning} --save_folder {folder name 1}
2. python get_matrix.py --read_folder {folder name 1} --save_folder {folder name 2}
```


### 3. Inference with Transform Matrix

```python
python inference_matrix.py  \
        --dataset XNLI \
        --model {path of the model after CC-Tuning} \
        --matrix_folder llama \
        --save_folder llama
```


## Citation
If you find our work useful, please cite the following paper~
```
@misc{ye2025cctuningcrosslingualconnectionmechanism,
      title={CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning}, 
      author={Yangfan Ye and Xiaocheng Feng and Zekun Yuan and Xiachong Feng and Libo Qin and Lei Huang and Weitao Ma and Yichong Huang and Zhirui Zhang and Yunfei Lu and Xiaohui Yan and Duyu Tang and Dandan Tu and Bing Qin},
      year={2025},
      eprint={2506.00875},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.00875}, 
}
```