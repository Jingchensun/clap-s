<div align="center">
<h1> CLAP-S: Support Set-Based Adaptation for Downstream Fiber-Optic Acoustic Recognition </h1>

[Jingchen Sun](https://jingchensun.github.io/)<sup>1, 2</sup>, [Shaobo Han](https://shaobohan.net/)<sup>1</sup>, [Wataru Kohno](https://scholar.google.es/citations?user=-o4nExgAAAAJ&hl=vi)<sup>1</sup>, [Changyou Chen](https://cse.buffalo.edu/~changyou/)<sup>2</sup>

<sup>1</sup>  NEC Labs America, USA  
<sup>2</sup>  University at Buffalo, The State University of New York, USA


[![GitHub Project](https://img.shields.io/badge/GitHub-Project-blue?logo=github)](https://github.com/Jingchensun/clap-s)
[![arXiv](https://img.shields.io/badge/arXiv-2501.09877-b31b1b.svg)](https://arxiv.org/abs/2501.09877) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Introduction
This repository contains the code and dataset for our paper `CLAP-S: Support Set-Based Adaptation for Downstream Fiber-Optic Acoustic Recognition`. [[Paper](https://arxiv.org/abs/2501.09877)]
![clap_diagrams](examples/main.png)

## Environment Installation
* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n claps python=3.10 -y

# Activate the environment
conda activate clap

# Clone this repository
git clone git@github.com:Jingchensun/clap-s.git
cd clap-s
# Install torch
pip install -r requirement.txt
```

## 1 Dataset Preparation
### Outdoor Fiber Acoustic Dataset

<div style="text-align: center;">
  <!-- <p><a href="examples/train_367.wav">audio sample: Crackers</a></p> -->
  <img src="examples/dataset.png" alt="audio sample: Crackers">
</div>

Download the outdoor gunshot-firework fiber acoustic dataset from [Google Drive](https://drive.google.com/drive/folders/1P45O415eiUN_Emp1ENTRkB5wcw2wCRos?ths=true) and extract them to $data$ folder.
```bash
tar -xvf coil_gunshot.tar.gz
tar -xvf fiber_gunshot.tar.gz
```
The Final data structure should looks like:
```
data/
|–– coil_gunshot/
|–– coil_gunshot_train_labels.txt
|–– coil_gunshot_test_labels.txt
|–– fiber_gunshot/
|–– fiber_gunshot_train_labels.txt
|–– fiber_gunshot_test_labels.txt
```


## 2 Run the CLAP-S
This command runs the CLAP-Support on the recorded ESC50 dataset. You can modify the dataset and random seed.

```bash
sh clap_support.sh 
```

## 3 Run the CLAP-S⁺ 
Before running CLAP-S⁺, you need to obtain the weights of the 'Adapter,' which consists of two MLP layers. Use the command below to train a few-shot Adapter and save the checkpoint.

```bash
sh clap_adapter.sh
```
After obtaining the Adapter checkpoint, you can train the trainable audio feature Adapter in CLAP-S. Note that this implicit Adapter is different from the explicit Adapter you trained earlier. This highlights a key insight of our method: leveraging the few-shot data twice.

```bash
sh clap_support_plus.sh
```
## 4 Experiment Result
### Fiber-Gunshot Performance

| Model  | Training Type       | 1-shot | 2-shot | 4-shot | 8-shot | Full shot |
|--------|---------------------|--------|--------|--------|--------|-----------|
| CLAP-S  | Training-Free      | 49.0   | 57.0   | 60.0   | 71.0   | 84.0      |
| CLAP-S⁺ | Training-Required  | 58.0   | 66.0   | 70.0   | 78.0   | 86.0      |

### Coil-Gunshot Performance

| Model  | Training Type       | 0-shot | 2-shot | 4-shot | 8-shot | Full shot |
|--------|---------------------|--------|--------|--------|--------|-----------|
| CLAP-S  | Training-Free      | 57.0   | 65.0   | 67.0   | 72.0   | 85.0      |
| CLAP-S⁺ | Training-Required  | 62.0   | 68.0   | 67.0   | 69.0   | 87.0      |

## Contact
If you have any questions regarding this repo, please contact Jingchen Sun (jsun39@buffalo.edu).

## Citing CLAP-S

If you find this repository useful, please consider giving a star :star: and citation

```
@article{sun2025clap,
  title={CLAP-S: Support Set Based Adaptation for Downstream Fiber-optic Acoustic Recognition},
  author={Sun, Jingchen and Han, Shaobo and Kohno, Wataru and Chen, Changyou},
  journal={arXiv preprint arXiv:2501.09877},
  year={2025}
}
```
