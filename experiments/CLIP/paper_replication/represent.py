import torch 
from model import CLIP
import pandas as pd
import numpy as np
import pickle, os, random
from PIL import Image
from utils import TextTokenizer, load_norvig_vocabulary
from torchvision import transforms

# load tokenizers and image transforms
vocabulary, vocab_size = load_norvig_vocabulary()
n_blocks = 6
tokenizer = TextTokenizer(vocabulary)
iamge_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# Configuration
images_path = "/home/agambetti/data/Yelp/NY/images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reviews = pd.read_csv("/home/agambetti/PhD/multi-media-interaction/data/raw/reviews.csv")

def subset_reviews(restaurant, sample=None):
    texts = reviews[reviews['Restaurant'] == restaurant].review.tolist()
    if sample:
        texts = random.choices(texts, k=sample)
    return texts

def subset_images(restaurant, sample=None):
    images = os.listdir(os.path.join(images_path, restaurant))
    images = [Image.open(os.path.join(images_path, restaurant, image)) for image in images]
    if sample:
        images = random.choices(images, k=sample)
    return images

def load_clip(path_to_model):
    model = CLIP(n_blocks, vocab_size)
    weights = torch.load(path_to_model)
    model.load_state_dict(weights)
    model = torch.compile(model)
    model.to(device)
    return model

def extract_embeddings(reviews, images):
    text_embeddings, image_embeddings = [], []
    for review, image in zip(reviews, images):
        text = tokenizer(review).to(device)
        image = iamge_transform(image).to(device)
        image_emb = model.project_image(image.unsqueeze(0))
        text_emb = model.project_text(text.unsqueeze(0))
        text_embeddings.append(text_emb)
        image_embeddings.append(image_emb)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    return text_embeddings, image_embeddings

def process_restaurant(restaurant, sample=None):
    reviews = subset_reviews(restaurant, sample)
    images = subset_images(restaurant, sample)
    if len(reviews) <= 30:
        text_emb, img_emb = extract_embeddings(reviews, images)
        text_emb, img_emb = text_emb.cpu().detach().numpy(), img_emb.cpu().detach().numpy()
        output = {'restaurant':restaurant, 'text':text_emb, 'image':img_emb}
    else:
        text_emb = []
        img_emb = []
        for i in range(0, len(reviews), 30):
            temp_text_emb, temp_img_emb = extract_embeddings(reviews[i:i+30], images[i:i+30])
            temp_text_emb, temp_img_emb = temp_text_emb.cpu().detach().numpy(), temp_img_emb.cpu().detach().numpy()
            text_emb.append(temp_text_emb), img_emb.append(temp_img_emb)
        text_emb = np.concatenate(text_emb, axis=0)
        img_emb = np.concatenate(img_emb, axis=0)
        output = {'restaurant':restaurant, 'text':text_emb, 'image':img_emb}
    return output

def main(sample=None, target=None):
    outputs = []
    print('Calculating embeddings..')
    for i, restaurant in enumerate(restaurants):
        if i % 100 == 0: print(f'Processing restaurant {i}..')
        try:
            output = process_restaurant(restaurant, sample)
            outputs.append(output)
        except Exception as e:
            print(e)
            continue
    
    # output 
    with open(f'multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/embeddings_{sample}.pkl', 'wb') as f:
        pickle.dump(outputs, f)
        print('Saved embeddings.')

if __name__ == "__main__":
    # Load the data
    sample = 10
    path_to_model = f"multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/weights-B16/clip_1.pt"
    path = f"./multi-media-interaction/data/restaurants/restaurants_{sample}.pkl"
    with open(path, 'rb') as f:
        restaurants = pickle.load(f)
        print(f'Total restaurants: {len(restaurants)}')
    print(f'Processing {sample}..')
    model = load_clip(path_to_model)
    main(sample=sample)
