# NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction
NeuroClips is a novel framework for high-quality and smooth fMRI-to-video reconstruction. 

## News
- Codes will be available in the next months
- Sep. 26, 2024. Accepted by NeurIPS 2024 for Oral Presentation. 
- May. 24, 2024. Project release.

## Abstract
Reconstruction of static visual stimuli from non-invasion brain activity fMRI achieves great success, owning to advanced deep learning models such as CLIP and Stable Diffusion. However, the research on fMRI-to-video reconstruction remains limited since decoding the spatiotemporal perception of continuous visual experiences is formidably challenging. We contend that the key to addressing these challenges lies in accurately decoding both high-level semantics and low-level perception flows, as perceived by the brain in response to video stimuli.
To the end, we propose NeuroClips, an innovative framework to decode high-fidelity and smooth video from fMRI. NeuroClips utilizes a semantics reconstructor to reconstruct video keyframes, guiding semantic accuracy and consistency, and employs a perception reconstructor to capture low-level perceptual details, ensuring video smoothness. During inference, it adopts a pre-trained T2V diffusion model injected with both keyframes and low-level perception flows for video reconstruction.
Evaluated on a publicly available fMRI-video dataset, NeuroClips achieves smooth high-fidelity video reconstruction of up to 6s at 8FPS, gaining significant improvements over state-of-the-art models in various metrics, e.g., a 128% improvement in SSIM and an 81% improvement in spatiotemporal metrics.

## Method
![model](assets/model.png)

## Reconstruction Demos
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
