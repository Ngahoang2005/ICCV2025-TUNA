# Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning

<div align="center">
  <div>
  <a href='https://www.lamda.nju.edu.cn/wangy/' target='_blank'>Yan Wang</a>&emsp;
    <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a>&emsp;
    <a href='http://www.lamda.nju.edu.cn/yehj' target='_blank'>Han-Jia Ye</a>&emsp;
    </div>
<div>
    State Key Laboratory for Novel Software Technology, Nanjing University
    </div>
</div>

<div align="center">

  <a href="https://arxiv.org/abs/2508.08165">
    <img src="https://img.shields.io/badge/Paper-red" alt="arXiv">
  </a>

</div>

The code repository for "[Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](https://arxiv.org/abs/2508.08165)" (ICCV 2025) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

```bibtex
@inproceedings{wang2025integrating,
  title={Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning},
  author={Yan Wang and Da-Wei Zhou and Han-Jia Ye},
  booktitle={ICCV},
  year={2025}
}
```

## 📢 **Updates**

[08/2025] Code has been released.

[08/2025] [arXiv](https://arxiv.org/abs/2508.08165) paper has been released.

[06/2025] Accepted to [ICCV 2025](https://iccv.thecvf.com/).

## 📝 Introduction

Class-Incremental Learning (CIL) requires a learning system to continually learn new classes without forgetting. Existing pre-trained model-based CIL methods often freeze the pre-trained network and adapt to incremental tasks using additional lightweight modules such as adapters. However, incorrect module selection during inference hurts performance, and task-specific modules often overlook shared general knowledge, leading to errors on distinguishing between similar classes across tasks. To address the aforementioned challenges, we propose integrating Task-Specific and Universal Adapters (TUNA) in this paper. Specifically, we train task-specific adapters to capture the most crucial features relevant to their respective tasks and introduce an entropy-based selection mechanism to choose the most suitable adapter. Furthermore, we leverage an adapter fusion strategy to construct a universal adapter, which encodes the most discriminative features shared across tasks. We combine task-specific and universal adapter predictions to harness both specialized and general knowledge during inference. Extensive experiments on various benchmark datasets demonstrate the state-of-the-art performance of our approach.

## Requirements

### 🗂️ Environment

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)

### 🔎 Dataset

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ImageNet-A**: Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) if the file is too large to download.

You need to modify the path of the datasets in `./utils/data.py` according to your own path.

> These datasets are referenced in the [Aper](https://github.com/zhoudw-zdw/RevisitingCIL)
> These datasets are referenced in the [Aper](https://github.com/zhoudw-zdw/RevisitingCIL)

### Kaggle: download ImageNet-R in one cell

Add the following cell near the top of your Kaggle notebook to fetch and unzip ImageNet-R before training:

```
!pip install --quiet gdown
!mkdir -p /kaggle/working/imagenet-r
!gdown --id 1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR -O /kaggle/working/imagenet-r.zip
!unzip -q /kaggle/working/imagenet-r.zip -d /kaggle/working
!mv /kaggle/working/imagenet-r /kaggle/working/imagenet-r || true
```

Set `imagenetr_root` to `/kaggle/working/imagenet-r` (already set in `exps/tuna_imgr.json`) or override it via `--imagenetr-root` when launching `main.py` so the dataloader can find the extracted folders.

### Saving and resuming long Kaggle runs

- Checkpoints are saved after each incremental task to `./checkpoints` by default (see `checkpoint_dir` and `save_checkpoints` in your config).
- To resume after a time-out, point `resume_checkpoint` (or `--resume-checkpoint` on the CLI) to the last saved `task_*.pkl` file. Training will continue from the next task while keeping exemplar memory and class statistics intact

## 🔑 Running scripts

Please follow the settings in the `exps` folder to prepare json files, and then run:

```
python main.py --config ./exps/[filename].json
```

## 👨‍🏫 Acknowledgment

We would like to express our gratitude to the following repositories for offering valuable components and functions that contributed to our work.

- [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox](https://github.com/sun-hailong/LAMDA-PILOT)
- [RevisitingCIL](https://github.com/zhoudw-zdw/RevisitingCIL)
