# NeuroClips:Towards High-fidelity and Smooth fMRI-to-Video Reconstruction
NeuroClips is a framework for high-quality and smooth fMRI-to-video reconstruction. 
## Abstract
Decoding visual stimuli is fundamental to the study of the human brain's visual system and even higher perceptual systems. Thanks to its good spatial resolution, decoding static images from fMRI has been a great success. However, the field of motion video reconstruction is still little explored. The great difficulty lies in the low-temporal resolution and signal-to-noise ratio of fMRI and the lack of pixel-level control. In this paper, we propose NeuroClips, an innovative framework to decode high-fidelity and smooth video from fMRI sequences. This framework utilizes perception reconstructor to capture the ambiguous dynamic processes of a scene, together with employing semantic reconstructor to reconstruct video keyframes and guarantee semantic accuracy and consistency. Evaluated on a publicly available fMRI-video dataset, NeuroClips achieves smooth video reconstruction of up to 6s at 8FPS, gaining significant improvements in many metrics, with a 128% improvement in SSIM and an 81% improvement in spatio-temporal metrics than previous state-of-the-art models.

## Method
![model](model.png)
