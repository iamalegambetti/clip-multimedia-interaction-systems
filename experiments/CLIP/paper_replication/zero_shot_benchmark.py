import torch 
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from dataset import RestaurantDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np 

# CONFIG
test_path = "multi-media-interaction/experiments/CLIP/paper_replication/data/test_reviews.csv"
images_path = "/home/agambetti/data/Yelp/NY/images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip = clip.to(device).eval()
clip = torch.compile(clip)

def extract_logits(model, reviews, images):
    inputs = processor(text = reviews, images=images, return_tensors="pt", padding=True, max_length=77, truncation=True).to(device)
    outputs = model(**inputs)
    logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
    return logits_per_image, logits_per_text

transform = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 16
dataset = RestaurantDataset(test_path, images_path, image_transforms=transform)
print(f'Dataset length: {len(dataset)}')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss()
losses = []

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        reviews = batch["review"]
        images = batch["image"]
        logits_per_image, logits_per_text = extract_logits(clip, reviews, images)
        labels = torch.arange(logits_per_image.shape[0]).to(device)
        loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2
        losses.append(loss.item())

print("Average loss:", np.mean(losses))