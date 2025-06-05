from huggingface_hub import snapshot_download, hf_hub_download

# Backbone from MindEye2
hf_hub_download(repo_id="pscotti/mindeyev2",subfolder='train_logs/final_subj01_pretrained_40sess_24bs',filename="last.pth",repo_type="dataset",cache_dir='./weights')
hf_hub_download(repo_id="pscotti/mindeyev2",filename="sd_image_var_autoenc.pth",repo_type="dataset",cache_dir='./weights')
hf_hub_download(repo_id="pscotti/mindeyev2",filename="convnext_xlarge_alpha0.75_fullckpt.pth",repo_type="dataset",cache_dir='./weights')



# Stable Diffusion v1.5
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='unet',filename="diffusion_pytorch_model.bin",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='unet',filename="config.json",cache_dir='./weights/')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='vae',filename="diffusion_pytorch_model.bin",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='vae',filename="config.json",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='text_encoder',filename="model.safetensors",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='text_encoder',filename="config.json",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="merges.txt",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="special_tokens_map.json",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="tokenizer_config.json",cache_dir='./weights')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="vocab.json",cache_dir='./weights')

# Animate Diffusion
hf_hub_download(repo_id="guoyww/animatediff",filename="v3_sd15_adapter",cache_dir='./weights')
hf_hub_download(repo_id="guoyww/animatediff",filename="v3_sd15_mm.ckpt",cache_dir='./weights')
hf_hub_download(repo_id="guoyww/animatediff",filename="v3_sd15_sparsectrl_rgb.ckpt",cache_dir='./weights')