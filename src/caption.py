from PIL import Image
from torchvision import transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir='/fs/scratch/PAS2490/blip2')
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", cache_dir='/fs/scratch/PAS2490/blip2', torch_dtype=torch.float16
)
model.to(device)


images = torch.load('/fs/scratch/PAS2490/neuroclips/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_test_3fps.pt',map_location='cpu')[:,2,:,:,:]
print(images.shape)
all_predcaptions = []
for i in range(images.shape[0]):
    x = images[i]
    x = transforms.ToPILImage()(x)
    inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    all_predcaptions = np.hstack((all_predcaptions, generated_text))
    print(generated_text, all_predcaptions.shape)

torch.save(all_predcaptions, f'/fs/scratch/PAS2490/neuroclips/GT_test_caption.pt')


images = torch.load('/fs/scratch/PAS2490/neuroclips/datasets--gongzx--cc2017_dataset/snapshots/a82b9e20e98710f18913a10c0a5bf5f19a6e4000/GT_train_3fps.pt',map_location='cpu')[:,2,:,:,:]
print(images.shape)
all_predcaptions = []
for i in range(images.shape[0]):
    x = images[i]
    x = transforms.ToPILImage()(x)
    inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    all_predcaptions = np.hstack((all_predcaptions, generated_text))
    print(generated_text, all_predcaptions.shape)

torch.save(all_predcaptions, f'/fs/scratch/PAS2490/neuroclips/GT_train_caption.pt')