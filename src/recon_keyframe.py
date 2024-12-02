import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils


### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
#device = accelerator.device
device = 'cuda:0'
print("device:",device)

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=4096,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs("evals",exist_ok=True)


voxels = {}
model_name =f'video_subj0{subj}_SR'

os.makedirs(f"/fs/scratch/PAS2490/neuroclips/frames_generated/{model_name}",exist_ok=True)
# Load hdf5 data for betas

if subj == 1:
    num_voxels = 13447
elif subj == 2:
    num_voxels = 14828
elif subj == 3:
    num_voxels = 9114

class CC2017_Dataset(torch.utils.data.Dataset):
    def __init__(self, voxel, image, istrain):
        if istrain == True:
            self.length = 4320
        else:
            self.length = 1200
        self.voxel = voxel
        self.image = image

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.voxel[idx], self.image[idx]

seq_len = 1

if subj == 1:
    voxel_length = 13447
elif subj == 2 :
    voxel_length = 14828
elif subj == 3 :
    voxel_length = 9114

voxel_test = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/subj0{subj}_train_fmri.pt', map_location='cpu')
voxel_test = torch.mean(voxel_test, dim = 1)
print("Loaded all fmri test frames to cpu!", voxel_test.shape)
test_images = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_test_3fps.pt',map_location='cpu')
print("Loaded all crucial test frames to cpu!", test_images.shape)

test_dataset = CC2017_Dataset(voxel_test, test_images, istrain = False)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=0, drop_last=False)

clip_seq_dim = 256
clip_emb_dim = 1664


if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'/users/PAS2490/marcusshen/.cache/huggingface/hub/datasets--pscotti--mindeyev2/snapshots/183269ab73b49d2fa10b5bfe077194992934e4e6/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
class Neuroclips(nn.Module):
    def __init__(self):
        super(Neuroclips, self).__init__()
    def forward(self, x):
        return x
        
model = Neuroclips()
class CLIPProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1664, 1280))
    def forward(self, x):
        x = torch.mean(x, dim = 1)
        x = x @ self.proj
        return x


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out

model.ridge = RidgeRegression([voxel_test.shape[-1]], out_features=hidden_dim, seq_len=seq_len)
model.clipproj = CLIPProj()


from Semantic import Semantic_Reconstruction
model.backbone = Semantic_Reconstruction(h=hidden_dim, in_dim=hidden_dim, seq_len=seq_len, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=False)
utils.count_params(model.backbone)
utils.count_params(model)



from Semantic import *

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb",
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
    
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)


print("\n---resuming from last.pth ckpt---\n")
checkpoint = torch.load(f'/fs/scratch/PAS2490/neuroclips/models/video_subj0{subj}_SR.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
del checkpoint

# setup text caption networks
from transformers import AutoProcessor
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024

class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x
        
clip_convert = CLIPConverter()
state_dict = torch.load(f"/fs/scratch/PAS2490/mindeye/weights/datasets--pscotti--mindeyev2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict

# prep unCLIP
config = OmegaConf.load("./generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)

ckpt_path = f'/fs/scratch/PAS2490/mindeye/weights/datasets--pscotti--mindeyev2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])
del ckpt

batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)

# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# all_images = None
all_blurryrecons = None
all_recons = None
all_predcaptions = []
all_clipvoxels = None
all_textvoxels = None

minibatch_size = 10
num_samples_per_image = 1
assert num_samples_per_image == 1
plotting = False
index = 0
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for voxel, image in test_dl:
        
        voxel = voxel.unsqueeze(1).to(device)
        voxel = voxel.half()
        
        voxel_ridge = model.ridge(voxel,0) # 0th index of subj_list
        backbone, clip_voxels, blurry_image_enc = model.backbone(voxel_ridge)
        backbone = clip_voxels

                
        # Save retrieval submodule outputs
        if all_clipvoxels is None:
            all_clipvoxels = clip_voxels.cpu()
        else:
            all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))
        
        # Feed voxels through OpenCLIP-bigG diffusion prior
        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)

        prior_out = prior_out.to(device)
        
        pred_caption_emb = clip_convert(prior_out)
        generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        print(generated_caption)
        
        # Feed diffusion prior outputs through unCLIP
        for i in range(len(voxel)):
            index += 1
            print(index)
            samples = utils.unclip_recon(prior_out[[i]],
                                diffusion_engine,
                                vector_suffix,
                                num_samples=num_samples_per_image,
                                device = device)
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))

            #transforms.ToPILImage()(samples[0]).save(f'/home/students/gzx_4090_1/Video/frames_generated/video_subj01_skiplora_text_40sess_10bs/images/{all_recons.shape[0]-1}.png')
            if plotting:
                for s in range(num_samples_per_image):
                    plt.figure(figsize=(2,2))
                    plt.imshow(transforms.ToPILImage()(samples[s]))
                    plt.axis('off')
                    plt.show()

            if blurry_recon:
                blurred_image = (autoenc.decode(blurry_image_enc[0]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                
                for i in range(len(voxel)):
                    im = torch.Tensor(blurred_image[i])
                    if all_blurryrecons is None:
                        all_blurryrecons = im[None].cpu()
                    else:
                        all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))
                    if plotting:
                        plt.figure(figsize=(2,2))
                        plt.imshow(transforms.ToPILImage()(im))
                        plt.axis('off')
                        plt.show()

            if plotting: 
                print(model_name)
                err # dont actually want to run the whole thing with plotting=True
            

# resize outputs before saving
imsize = 256
all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
if blurry_recon: 
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()
        
# saving
print(all_recons.shape)
# torch.save(all_images,"evals/all_images.pt")
if blurry_recon:
    torch.save(all_blurryrecons,f"/fs/scratch/PAS2490/neuroclips/frames_generated/{model_name}/{model_name}_all_blurryrecons.pt")
torch.save(all_recons,f"/fs/scratch/PAS2490/neuroclips/frames_generated/{model_name}/{model_name}_all_recons.pt")
torch.save(all_predcaptions,f"/fs/scratch/PAS2490/neuroclips/frames_generated/{model_name}/{model_name}_all_predcaptions.pt")
torch.save(all_clipvoxels,f"/fs/scratch/PAS2490/neuroclips/frames_generated/{model_name}/{model_name}_all_clipvoxels.pt")
print(f"saved {model_name} outputs!")

if not utils.is_interactive():
    sys.exit(0)