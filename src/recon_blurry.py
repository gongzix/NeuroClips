import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

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

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")

print("PID of this process =",os.getpid())
device = accelerator.device
#device = 'cuda:0'
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3],
    help="Validate on which subject?",
)
parser.add_argument(
    "--use_prior",action=argparse.BooleanOptionalAction,default=False,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--batch_size", type=int, default=10,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--blur_scale",type=float,default=.5,
    help="multiply loss from blurry recons by this number",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--prior_scale",type=float,default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=150,
    help="number of epochs of training",
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=4096,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--use_text",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--fps",type=int,default=3,
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
model_name =f'video_subj0{subj}_lPR'

outdir = os.path.abspath(f'/fs/scratch/PAS2490/neuroclips/{model_name}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
    

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


voxel_test = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/subj0{subj}_test_fmri.pt', map_location='cpu')
voxel_test = torch.mean(voxel_test, dim = 1).unsqueeze(1)
print("Loaded all fmri test frames to cpu!", voxel_test.shape)
num_voxels_list = [voxel_test.shape[-1]]

test_images = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_test_3fps.pt',map_location='cpu')

print("Loaded all crucial test frames to cpu!", test_images.shape)


test_dataset = CC2017_Dataset(voxel_test, test_images, istrain = False)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

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
    ckpt = torch.load(f'/fs/scratch/PAS2490/mindeye/weights/datasets--pscotti--mindeyev2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d/sd_image_var_autoenc.pth')
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

model.ridge1 = RidgeRegression(num_voxels_list, out_features=hidden_dim, seq_len=seq_len)
utils.count_params(model.ridge1)
utils.count_params(model)


from Perception import Perception_Reconstruction, Inception_Extension
model.backbone = Perception_Reconstruction(h=hidden_dim, in_dim=hidden_dim, seq_len=seq_len, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale)
model.fmri = Inception_Extension(h=256, in_dim=voxel_length, out_dim=voxel_length, expand=fps*2, seq_len=seq_len)
utils.count_params(model.backbone)
utils.count_params(model.fmri)
utils.count_params(model)


print("\n---resuming from last.pth ckpt---\n")
checkpoint = torch.load(f'/fs/scratch/PAS2490/neuroclips/models/video_subj0{subj}_PR.pth', map_location='cpu')['model_state_dict']
model.load_state_dict(checkpoint, strict=True)
del checkpoint


epoch = 0
torch.cuda.empty_cache()

model = accelerator.prepare(model)
print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
model.eval()
all_blurryrecons = None

if local_rank==0:
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
        for test_i, (voxels, images) in enumerate(test_dl):  
            # all test samples should be loaded per batch such that test_i should never exceed 0

            ## Average same-image repeats ##
            if test_image is None:
                voxel = voxels.half()
                image = images.cpu()

            loss=0.
                        
            voxel = voxel.to(device)
            image = image.to(device)

            voxel = model.fmri(voxel).unsqueeze(1)
            voxel_ridge = model.ridge1(voxel,0)
            blurry_image_enc_ = model.backbone(voxel_ridge, time= batch_size)  
            
            if blurry_recon:
                image_enc_pred, _ = blurry_image_enc_
                blurry_recon_images = (autoenc.decode(image_enc_pred/0.18215).sample / 2 + 0.5).clamp(0,1)

            for i in range(len(voxel)):
                im = torch.Tensor(blurry_recon_images[i])
                if all_blurryrecons is None:
                    all_blurryrecons = im[None].cpu()
                else:
                    all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))
            
            print(all_blurryrecons.shape)
    torch.save(all_blurryrecons, outdir+f'/{model_name}_PR.pt')