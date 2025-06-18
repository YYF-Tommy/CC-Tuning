import numpy as np
import os
from tqdm.auto import tqdm
import random
import argparse


def MSE(A, B):
    # 计算 MSE
    print(A, B)
    mse = np.mean((A - B) ** 2)
    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_folder", type=str, default="Llama-3.1-8B")
    parser.add_argument("--save_folder", type=str, default="Llama-3.1-8B")
    args = parser.parse_args()

    read_folder = args.read_folder
    save_folder = args.save_folder

    ########## Step 1
    source_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{read_folder}/nonEn"
    target_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{read_folder}/En"

    source_files = os.listdir(source_path)
    target_files = os.listdir(target_path)

    print(len(source_files), len(target_files))
    print(np.load(f"{source_path}/{source_files[0]}").shape)
    print(np.load(f"{target_path}/{target_files[0]}").shape)

    source_vectors = []
    target_vectors = []

    for s, t in tqdm(zip(source_files, target_files)):
        tmp_s = np.load(f"{source_path}/{s}")
        tmp_t = np.load(f"{target_path}/{t}")
        source_vectors.append(tmp_s)
        target_vectors.append(tmp_t)

    source_vectors = np.stack(source_vectors)
    target_vectors = np.stack(target_vectors)
    print(source_vectors.shape)
    print(target_vectors.shape)

    if len(source_vectors.shape) == 1:
        source_vectors = source_vectors.reshape(1, source_vectors.shape[0])
        target_vectors = target_vectors.reshape(1, target_vectors.shape[0])

    np.save(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{read_folder}/nonEn.npy", source_vectors)
    np.save(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{read_folder}/En.npy", target_vectors)


    ########## Step 2
    source_vectors = np.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{read_folder}/nonEn.npy")
    target_vectors = np.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/cached_vectors/{read_folder}/En.npy")

    print(source_vectors.shape)
    print(target_vectors.shape)

    source_vectors = source_vectors.reshape((source_vectors.shape[0]*source_vectors.shape[1], source_vectors.shape[2]))
    target_vectors = target_vectors.reshape((target_vectors.shape[0]*target_vectors.shape[1], target_vectors.shape[2]))

    assert(source_vectors.shape == target_vectors.shape)

    # representations1 -> representations2
    # X * source_vectors = target_vectors  ==>  X = target_vectors * source_vectors.T * (source_vectors * source_vectors.T)^(-1)

    # 使用最小二乘法 (A * X = B)
    X, _, _, _ = np.linalg.lstsq(source_vectors, target_vectors, rcond=None)

    # 保存映射矩阵
    save_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/TransformMatrix/saved_matrix"
    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/{read_folder}.npy", X)

    print(MSE(target_vectors, source_vectors @ X))
