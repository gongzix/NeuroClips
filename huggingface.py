from huggingface_hub import snapshot_download, hf_hub_download

# Stable Diffusion v1.5
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='unet',filename="diffusion_pytorch_model.bin",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='unet',filename="config.json",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='vae',filename="diffusion_pytorch_model.bin",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='vae',filename="config.json",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='text_encoder',filename="model.safetensors",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='text_encoder',filename="config.json",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="merges.txt",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="special_tokens_map.json",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="tokenizer_config.json",cache_dir='./')
hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",subfolder='tokenizer',filename="vocab.json",cache_dir='./')

# Animate Diffusion
hf_hub_download(repo_id="guoyww/animatediff",filename="v3_sd15_adapter",cache_dir='./Animatediff/models')
hf_hub_download(repo_id="guoyww/animatediff",filename="v3_sd15_mm.ckpt",cache_dir='./Animatediff/models')
hf_hub_download(repo_id="guoyww/animatediff",filename="v3_sd15_sparsectrl_rgb.ckpt",cache_dir='./Animatediff/models')