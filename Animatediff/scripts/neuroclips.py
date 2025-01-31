import argparse
import datetime
import inspect
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_neuroclips import NeuroclipsPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange

from pathlib import Path
from PIL import Image

def cccat(A):

    output_tensors = []
    output_tensors.append(A[:,0].unsqueeze(1))
    for i in range(A.size(1)-1):
        
        output_tensors.append((0.67*A[:,i]+0.33*A[:,i+1]).unsqueeze(1))
        output_tensors.append((0.33*A[:,i]+0.67*A[:,i+1]).unsqueeze(1))
        output_tensors.append(A[:,i+1].unsqueeze(1))

    output_tensor = torch.cat(output_tensors, dim=1)
    return output_tensor

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    #os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = NeuroclipsPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        subj = args.subj
        # Your saved path of the reconstructed blurry video
        blurry = torch.load(f'/fs/scratch/PAS2490/neuroclips/blurry/video_subj0{subj}_PR/video_subj0{subj}_PR.pt', map_location='cpu').reshape(1200*6,3,224,224).float()
        # Your saved path of the caption of reconstructed keyframes
        caption = torch.load(f'/fs/scratch/PAS2490/neuroclips/frames_generated/video_subj0{subj}_SR/video_subj0{subj}_blip_caption.pt', map_location='cpu')
        # Your saved path of the reconstructed keyframes
        keyframes    = torch.tensor(torch.load(f'/fs/scratch/PAS2490/neuroclips/frames_generated/video_subj0{subj}_SR/video_subj0{subj}_SR_all_all_recons.pt',map_location='cpu'))
        blurry = transforms.Resize((args.W,args.H))(blurry).float()
        blurry = blurry.reshape(1200,6,3,256,256)
        keyframes = transforms.Resize((args.W,args.H))(keyframes).float()
        print(blurry.shape)
        print(keyframes.shape)
        random_seeds = 12099779162349365895
        n_prompt = "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        all_recon_video = None
        batch_size = 1
        for i in range(0, blurry.shape[0], batch_size):
            
            # manually set random seed for reproduction
            '''
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            '''

            torch.manual_seed(random_seeds)
            text = caption[i:i + batch_size]  
            prompt = [s+', 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3' for s in text]   
            print(f"sampling {prompt} ...")
            motion = blurry[i:i + batch_size].cuda()
            motion = transforms.Resize((args.W,args.H))(motion).float()
            motion = cccat(motion)
            keyframe = keyframes[i:i + batch_size].repeat(model_config.L,1,1,1).cuda()
            motion = rearrange(motion, "b f c h w -> (b f) c h w")
            #keyframe = rearrange(keyframe, "b f c h w -> (b f) c h w")
            latents = (vae.encode(2*motion-1).latent_dist.sample()* 0.18215)
            keylatents = (vae.encode(2*keyframe-1).latent_dist.sample()* 0.18215)
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b = batch_size)
            keylatents = rearrange(keylatents, "(b f) c h w -> b c f h w", b = batch_size)

            if model_config.get("controlnet_path", "") != "":
                # Your saved path of the first-frame, you can use the keyframe, but needs jpg/png
                image_path = [(f'/fs/scratch/PAS2490/neuroclips/frames_generated/video_subj0{subj}_SR/images/'+str(n)+'.png') for n in range(i, i + batch_size)]
                controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_path]

                controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
                controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

                if controlnet.use_simplified_condition_embedding:
                    #num_controlnet_images = controlnet_images.shape[2]
                    num_controlnet_images = 1
                    controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                    controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                    controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,
                low_strength        = 0.3,
                latents             = latents,
                keylatents          = keylatents,

                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
            ).videos
            #samples.append(sample)
            if all_recon_video is None:
                all_recon_video = sample.cpu()
            else:
                all_recon_video = torch.vstack((all_recon_video, sample.cpu()))
            print('subj',args.subj,all_recon_video.shape)
            # Save gif file
            save_videos_grid(sample[:,:,4:,:,:], f"/users/PAS2490/marcusshen/fMRIVideo_Nips/AnimateDiff/scripts/results/subj0{subj}/{i}.gif",fps=6)
            
            sample_idx += 1
    # Save all of the reconstructed videos for evaluation
    torch.save(all_recon_video,f'/fs/scratch/PAS2490/mindeye/video_generated/subj0{subj}/subj0{subj}_video.pt')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Your saved StableDiffusion v1.5 path, from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main, 
    this code needs text_encoder, tokenizer, unet, vae. .
    '''
    parser.add_argument("--pretrained-model-path", type=str, default="/users/PAS2490/marcusshen/fMRIVideo_Nips/AnimateDiff/models/StableDiffusion/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--subj", type=int, default=1)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
