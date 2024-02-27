import torch 
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import pickle, os, random
from PIL import Image

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
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16") # https://huggingface.co/openai/clip-vit-base-patch16
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16") 
    weights = torch.load(path_to_model)
    model.load_state_dict(weights)
    model = torch.compile(model)
    model.to(device)
    return model, processor 

def extract_embeddings(reviews, images):
    inputs = processor(text = reviews, images=images, return_tensors="pt", padding='max_length', max_length=77, truncation=True).to(device)
    outputs = model(**inputs)
    text_embedding = outputs.text_embeds
    image_embedding = outputs.image_embeds
    return text_embedding, image_embedding

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
    with open(f'multi-media-interaction/experiments/rq2_restaurants/CLIP/fine_tuned_embeddings/fine_tuned_{sample}.pkl', 'wb') as f:
        pickle.dump(outputs, f)
        print('Saved embeddings.')

if __name__ == "__main__":
    # Load the data
    sample = 10
    path_to_model = f"multi-media-interaction/experiments/rq2_restaurants/CLIP/fine_tuned_embeddings/weights_ft/clip_1.pt"
    path = f"./multi-media-interaction/data/restaurants/restaurants_{sample}.pkl"
    with open(path, 'rb') as f:
        restaurants = pickle.load(f)
        print(f'Total restaurants: {len(restaurants)}')
    print(f'Processing {sample}..')
    model, processor = load_clip(path_to_model)
    main(sample=sample)
