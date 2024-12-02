<h3 align="center"> NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction <div align=center> </a></h3>

## <h5 align="center"> NeuroClips is a novel framework for fMRI-to-video decoding (NeurIPS 2024 Oral). If you like our project, please give us a star ⭐.</h5>

<h5 align="center">

[![Huggingface NeuroClips](https://img.shields.io/static/v1?label=Model&message=Huggingface&color=orange)](https://huggingface.co/datasets/gongzx/cc2017_dataset/) &ensp;
[![arxiv](https://img.shields.io/badge/Arxiv-2410.19452-red)](https://arxiv.org/pdf/2410.19452) &ensp;
![GitHub Repo stars](https://img.shields.io/github/stars/gongzix/NeuroClips)
</h5>

## 🛠️ Method
![model](assets/model.png)

## 📣 News
- Training Codes will be available in this week.
- Nov. 30, 2024. Pre-processed code and dataset release.
- Sep. 26, 2024. Accepted by NeurIPS 2024 for Oral Presentation. 
- May. 24, 2024. Project release.

## Data Preprocessing
We use the public cc2017(Wen) dataset from [this](https://purr.purdue.edu/publications/2809/1). You can download and follow the [official preprocess](./preprocess/preprocess_code/ReadMe.pdf) to only deal with your fMRI data. Only use `movie_fmri_data_processing.m` and `movie_fmri_reproducibility.m`, and notice that the selected voxels(Bonferroni correction, P < 0.05) were more than before(Bonferroni correction, P < 0.01).

We also offer our pre-processed fMRI data and frames sampled from videos for training in NeuroClips, and you can directly download them from [![Huggingface NeuroClips](https://img.shields.io/static/v1?label=Dataset&message=Huggingface&color=orange)](https://huggingface.co/datasets/gongzx/cc2017_dataset/)

## Installation
Due to the conflict issue of different environment package versions, we recommend using the virtual environment for Neuroclips training, inference keyframes, and blurry videos separately from the pre-trained T2V diffusion's virtual environment.
For Neuroclips:
```python
. src/setup.sh
```
For pre-trained AnimateDiffusion, you can follow [this](https://github.com/guoyww/AnimateDiff).
## Training_SR
We suggest training the backbone first and then the prior to achieve better Semantic Reconstructor.

❗Warning: BLIP-2 isn't available now for the latest version in transformers==v4.47.dev. You can see the latest news in [this](https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/). Thus, we have excluded the text enhancement here. Once available, we will make corrections.
```python
python src/train_SR.py --subj 1 --batch_size 240 --num_epochs 30 --mixup_pct 1.0 --max_lr 1e-4
python src/train_SR.py --subj 1 --batch_size 64 --num_epochs 150 --mixup_pct 0.0 --max_lr 3e-4 --use_prior
```

## Training_PR
```python
python src/train_PR.py --subj 1 --batch_size 40 --mixup_pct 0.0 --num_epochs 80
```

## Reconstruct Keyframe
```python
python src/recon_keyframe.py --subj 1
```

## 🖼️ Reconstruction Demos
### *Human Behavior*
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_22.gif"></td>
      <td style="border: none"><img src="assets/samples/22.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_11.gif"></td>
      <td style="border: none"><img src="assets/samples/11.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_54.gif"></td>
      <td style="border: none"><img src="assets/samples/54.gif"></td>
      </tr>
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_936.gif"></td>
      <td style="border: none"><img src="assets/samples/936.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_373.gif"></td>
      <td style="border: none"><img src="assets/samples/373.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_769.gif"></td>
      <td style="border: none"><img src="assets/samples/769.gif"></td>
      </tr>
  </table>

### *Animals*
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_94.gif"></td>
      <td style="border: none"><img src="assets/samples/94.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_108.gif"></td>
      <td style="border: none"><img src="assets/samples/108.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_452.gif"></td>
      <td style="border: none"><img src="assets/samples/452.gif"></td>
      </tr>
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_293.gif"></td>
      <td style="border: none"><img src="assets/samples/293.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_784.gif"></td>
      <td style="border: none"><img src="assets/samples/784.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_216.gif"></td>
      <td style="border: none"><img src="assets/samples/216.gif"></td>
      </tr>
  </table>

### *Traffic*
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_24.gif"></td>
      <td style="border: none"><img src="assets/samples/24.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_555.gif"></td>
      <td style="border: none"><img src="assets/samples/555.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_180.gif"></td>
      <td style="border: none"><img src="assets/samples/180.gif"></td>
  </table>

### *Natural Scene*
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_700.gif"></td>
      <td style="border: none"><img src="assets/samples/700.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_702.gif"></td>
      <td style="border: none"><img src="assets/samples/702.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_284.gif"></td>
      <td style="border: none"><img src="assets/samples/284.gif"></td>
  </table>


### *Multi-fMRI Fusion*
With the help of NeuroClips’ SR, we explored the generation of longer videos for the first time. Since the technical field of long video generation is still immature, we chose a more straightforward fusion strategy that does not require additional GPU training. In the inference process, we consider the semantic similarity of two reconstructed keyframes from two neighboring fMRI samples (here we directly determine whether they belong to the same class of objects, e.g., both are jellyfish). If semantically similar, we replace the keyframe of the latter fMRI with the tail-frame of the former fMRI’s reconstructed video, which will be taken as the first-frame of the latter fMRI to generate the video.

![fusion](assets/samples/multi-fmri.gif)

## Fail Cases
Overall the fail cases can be divided into two categories: on the one hand, the semantics are not accurate enough and on the other hand, the scene transition affects the generated results.
### *Pixel Control & Semantic Deficit*
In CC2017 dataset, the video clips in the testing movie were different from those in the training movie, and there were even some categories of objects that didn't appear in the training set. However thanks to NeuroClips' Perceptual Reconstructor, we can still reconstruct the video at a low-level of vision.
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_99.gif"></td>
      <td style="border: none"><img src="assets/samples/99.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_232.gif"></td>
      <td style="border: none"><img src="assets/samples/232.gif"></td>
  </table>

### *Scene Transitions*
Due to the low-temporal resolution of fMRI (i.e., 2s), a segment of fMRI may include two video scenes, leading to semantic confusion in the video reconstruction, or even semantic and perceptual fusion, as shown in the following image of a jellyfish transitioning to the moon, which ultimately generates a jellyfish with a black background.
<table class="center">
      <tr style="line-height: 0">
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      <td colspan="1" style="border: none; text-align: center">GT</td> <td colspan="1" style="border: none; text-align: center">Ours</td>
      </tr>
      <td style="border: none"><img src="assets/samples/gt_97.gif"></td>
      <td style="border: none"><img src="assets/samples/97.gif"></td>
      <td style="border: none"><img src="assets/samples/gt_281.gif"></td>
      <td style="border: none"><img src="assets/samples/281.gif"></td>
  </table>

## BibTeX
```python
@article{gong2024neuroclips,
  title={NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction},
  author={Gong, Zixuan and Bao, Guangyin and Zhang, Qi and Wan, Zhongwei and Miao, Duoqian and Wang, Shoujin and Zhu, Lei and Wang, Changwei and Xu, Rongtao and Hu, Liang and others},
  journal={arXiv preprint arXiv:2410.19452},
  year={2024}
}
```

## Acknowledgement
We sincerely thank the following authors, and Neuroclips is based on their excellent open-source projects or impressive ideas.

T2V diffusion: https://github.com/guoyww/AnimateDiff

Excellent Backbone: https://github.com/MedARC-AI/MindEyeV2

Temporal Design: https://arxiv.org/abs/2304.08818

Keyframe Captioning: https://github.com/salesforce/LAVIS/tree/main/projects/blip2

Dataset and Pre-processed code: https://purr.purdue.edu/publications/2809