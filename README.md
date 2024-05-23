# NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction
NeuroClips is a novel framework for high-quality and smooth fMRI-to-video reconstruction. 
## Abstract
Reconstruction of static visual stimuli from non-invasion brain activity fMRI achieves great success, owning to advanced deep learning models such as CLIP and Stable Diffusion. However, the research on fMRI-to-video reconstruction remains limited since decoding the spatiotemporal perception of continuous visual experiences is formidably challenging. We contend that the key to addressing these challenges lies in accurately decoding both high-level semantics and low-level perception flows, as perceived by the brain in response to video stimuli.
To the end, we propose NeuroClips, an innovative framework to decode high-fidelity and smooth video from fMRI. NeuroClips utilizes a semantics reconstructor to reconstruct video keyframes, guiding semantic accuracy and consistency, and employs a perception reconstructor to capture low-level perceptual details, ensuring video smoothness. During inference, it adopts a pre-trained T2V diffusion model injected with both keyframes and low-level perception flows for video reconstruction.
Evaluated on a publicly available fMRI-video dataset, NeuroClips achieves smooth high-fidelity video reconstruction of up to 6s at 8FPS, gaining significant improvements over state-of-the-art models in various metrics, e.g., a 128% improvement in SSIM and an 81% improvement in spatiotemporal metrics.

## Method
![model](assets/model.png)

##
<table class="center">
      <tr style="line-height: 0">
      <td colspan="2" style="border: none; text-align: center">GT</td>
      <td colspan="2" style="border: none; text-align: center">GT</td>
      <td colspan="2" style="border: none; text-align: center">GT</td>
      <td colspan="2" style="border: none; text-align: center">GT</td>
      </tr>
      <tr>
      <td style="border: none"><img src="samples/0-a-car-is-driving-down-the-road,-8k-uhd,-dslr,.gif"></td>
      <td style="border: none"><img src="samples/93.gif"></td>
      <td style="border: none"><img src="samples/94.gif"></td>
      <td style="border: none"><img src="samples/95.gif"></td>
      <td style="border: none"><img src="samples/96.gif"></td>
      <td style="border: none"><img src="samples/108.gif"></td>
      <td style="border: none"><img src="samples/204.gif"></td>
      <td style="border: none"><img src="samples/281.gif"></td>
      </tr>
      <tr style="line-height: 0">
      <td colspan="2" style="border: none; text-align: center">GT</td>
      <td colspan="2" style="border: none; text-align: center">GT</td>
      <td colspan="2" style="border: none; text-align: center">GT</td>
      <td colspan="2" style="border: none; text-align: center">GT</td>
      </tr>
      <tr>
      <td style="border: none"><img src="samples/0-a-car-is-driving-down-the-road,-8k-uhd,-dslr,.gif"></td>
      <td style="border: none"><img src="samples/93.gif"></td>
      <td style="border: none"><img src="samples/94.gif"></td>
      <td style="border: none"><img src="samples/95.gif"></td>
      <td style="border: none"><img src="samples/96.gif"></td>
      <td style="border: none"><img src="samples/108.gif"></td>
      <td style="border: none"><img src="samples/204.gif"></td>
      <td style="border: none"><img src="samples/281.gif"></td>
      </tr>
  </table>

