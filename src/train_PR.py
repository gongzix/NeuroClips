import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import gc

import matplotlib.pyplot as plt
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
model_name =f'video_subj0{subj}_low_level'

outdir = os.path.abspath(f'/fs/scratch/PAS2490/neuroclips/models/')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
    
if use_image_aug or blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )
    

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

num_samples_per_epoch = (4320) // num_devices 
num_iterations_per_epoch = num_samples_per_epoch // (batch_size)
print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)

subj_list = [subj]
seq_len = 1


if subj == 1:
    voxel_length = 13447
elif subj == 2 :
    voxel_length = 14828
elif subj == 3 :
    voxel_length = 9114

voxel_train = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/subj0{subj}_train_fmri.pt', map_location='cpu')
voxel_test = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/subj0{subj}_test_fmri.pt', map_location='cpu')
## Average same-video repeats ##
voxel_test = torch.mean(voxel_test, dim = 1).unsqueeze(1)
num_voxels_list = [voxel_train.shape[-1]]

train_images = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000//GT_train_3fps.pt',map_location='cpu')
test_images = torch.load(f'/fs/scratch/PAS2490/neuroclips/voxel_mask/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_test_3fps.pt',map_location='cpu')

print("Loaded all crucial train frames to cpu!", train_images.shape)
print("Loaded all crucial test frames to cpu!", test_images.shape)


train_dl = {}
train_dataset = CC2017_Dataset(voxel_train, train_images, istrain = True)
test_dataset = CC2017_Dataset(voxel_test, test_images, istrain = False)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=0, drop_last=False)

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
    ckpt = torch.load(f'/fs/scratch/PAS2490/neuroclips/weights/datasets--pscotti--neuroclipsv2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
    from autoencoder.convnext import ConvnextXL
    
    cnx = ConvnextXL(f'/fs/scratch/PAS2490/neuroclips/weights/datasets--pscotti--neuroclipsv2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)
    
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)
    
    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
        data_keys=["input"],
    )

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

# test on subject 1 with fake data


for param in model.parameters():
    param.requires_grad_(True)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)

epoch = 0
losses, test_losses, lrs = [], [], []
best_test = 0
torch.cuda.empty_cache()
model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals



print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
text_scale = 0.3
if num_devices!=0 and distributed:
    model = model.module

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    
    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.
    
    loss_prior_total = 0.
    test_loss_prior_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. 
    
    # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
    step = 0
    for train_i, (voxel, image) in enumerate(train_dl):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss=0.

            #text = text_iters[train_i].detach()
            image = image.reshape(len(image)*fps*2, 3, 224, 224).to(device)
            voxel = voxel[:,epoch%2,:].unsqueeze(1).half().to(device)

            if use_image_aug: 
                image = img_augment(image)

            voxel = model.fmri(voxel).unsqueeze(1)
            voxel_ridge = model.ridge1(voxel,0)
            blurry_image_enc_ = model.backbone(voxel_ridge, time = batch_size*fps*2)

            if blurry_recon:     
                image_enc_pred, transformer_feats = blurry_image_enc_

                image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()

                if epoch < int(mixup_pct * num_epochs):
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                image_norm = (image - mean)/std
                image_aug = (blur_augs(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.2)
                loss_blurry_cont_total += cont_loss.item()

                loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215

            if blurry_recon:
                with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image), replace=False)
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()
            step += 1
            print(f'Training epoch: {epoch}, sample: {step*batch_size}, lr: {optimizer.param_groups[0]["lr"]}, loss: {loss.item():.4f}, loss_mean: {np.mean(losses[-(train_i+1):]):.4f}')
    
    model.eval()
    
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (voxel, image) in enumerate(test_dl):  
                # all test samples should be loaded per batch such that test_i should never exceed 0

                if test_image is None:
                    voxel = voxel.half()
                    
                    image = image.reshape(len(image)*fps*2, 3, 256, 256).cpu()

                loss=0.
                            
                voxel = voxel.to(device)
                image = image.to(device)

                #clip_target = clip_img_embedder(image.float())

                test_fwd_percent_correct = 0.
                test_bwd_percent_correct = 0.
                text_fwd_percent_correct = 0.

                voxel = model.fmri(voxel).unsqueeze(1)
                voxel_ridge = model.ridge1(voxel,0)
                blurry_image_enc_ = model.backbone(voxel_ridge, time= 40*fps*2)
               
                # for some evals, only doing a subset of the samples per batch because of computational cost
                #random_samps = np.random.choice(np.arange(len(image)), size=len(image)//6, replace=False)     
                
                if blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image, blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item() 

                    print('PixCorr:', pixcorr.item())

                #utils.check_loss(loss)              
                    test_losses.append(pixcorr.item())

            # if utils.is_interactive(): clear_output(wait=True)
            print("-------------------------")
    
    # Save model checkpoint and reconstruct
    if test_blurry_pixcorr/30 > best_test:
        best_test = test_blurry_pixcorr/30
        print('new best test loss:',best_test)
        save_ckpt(f'{model_name}')
    else:
        print('not best',test_blurry_pixcorr/30,'best test loss is',best_test)

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()

print("\n===Finished!===\n")
#if ckpt_saving:
    #save_ckpt(f'{model_name}')