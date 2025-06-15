[![arXiv](https://img.shields.io/badge/arXiv-2506.10973-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2506.10973)

# Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning

![intro](https://github.com/user-attachments/assets/cc067d12-c37e-48aa-a54c-c2101762a831)

This repo contains the code to reproduce the experiments from the paper ["Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning."](https://arxiv.org/abs/2506.10973)

## Usage 
The configurations for our runs, including all hyperparameters, can be found in `configs/`. Models that are not in the Neural Operator library can be found in `models/`. There are three modes that we consider in this paper: single-resolution training, multi-resolution training, and input-output interpolation.

For single resolution or multi-resolution training, run `train_single_res.py` and `train_multi_res.py` (respectively) as
```
python [single or multi res.py] [name of config file]
```
For example, to train FNO in single resolution, run `python train_single_res.py fno.yaml`. To train U-net in multiple resolutions, run `python train_multi_res.py unet.yaml`.

Input-output interpolation is a setting where a pre-trained model is evaluated by interpolating the input and the output to the model's training resolution. To evaluate a model's performance at other data resolutions using input-output interpolation, run
```
python evaluate_input_output_interpolation.py [name of weights file] [config name] [interpolation mode]
```
where `[name of weights file]` is the name of the model's pretrained weights in the directory `ckpts/`, and `[interpolation mode]` is one of `'fourier'` or `'bilinear'`.


## Requirements
This repo requires an installation of the [Neural Operator library](https://github.com/neuraloperator/neuraloperator/) (verified on [this commit](https://github.com/neuraloperator/neuraloperator/tree/d8c9b30fd72359e60a13397b72e92ca13b66a453) from 28 May, 2025) and [HuggingFace Transformers](https://github.com/huggingface/transformers), as well as both of their dependencies. The experiments in this paper focus on the Navier-Stokes equations, and the dataset we used can be [downloaded from Zenodo](https://zenodo.org/records/12825163).

## Citation
If you used our code in an academic publication, please cite the following:
```
@article{berner2025principled,
  title={Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning},
  author={Berner, Julius and Liu-Schiaffini, Miguel and Kossaifi, Jean and Duruisseaux, Valentin and Bonev, Boris and Azizzadenesheli, Kamyar and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2506.10973},
  year={2025}
}
```
