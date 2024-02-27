from model import CLIP 
from PIL import Image
import pandas as pd 
from utils import TextTokenizer, load_norvig_vocabulary
from torchvision import transforms
import os, random, torch 
from sklearn.metrics import accuracy_score
import numpy as np 
random.seed(46)

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

df = pd.read_csv("/home/agambetti/PhD/multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/data/test_reviews.csv")
images_path = "/home/agambetti/data/Yelp/NY/images"
weights_path = "/home/agambetti/PhD/multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/weights-B16/clip_1.pt"

def list_images(restaurant, reviwer):
    images = os.listdir(os.path.join(images_path, restaurant))
    images = [image for image in images if reviwer in image]
    return images

df['images'] = df.apply(lambda x: list_images(x.Restaurant, x.Reviewer), axis=1)
df['image'] = df.apply(lambda x: random.choice(x.images) if len(x.images) > 0 else None, axis=1)
df['image_path'] = df.apply(lambda x: os.path.join(images_path, x.Restaurant, x.image) if x.image else None, axis=1)

def sample_n_rest(n):
    # rewrite sample_pair to sample N restaurant
    restaurants = df.Restaurant.drop_duplicates().sample(n).tolist()
    output = []
    for i, restaurant in enumerate(restaurants):
        temp = df[df.Restaurant == restaurant].sample(1, random_state=46)
        temp = temp[['review', 'image_path']]
        temp['label'] = i
        output.append(temp)
    return pd.concat(output)

def calculate_accuracy(logits, labels):
    predicted_labels = torch.argmax(logits, dim=1).tolist()
    return accuracy_score(labels, predicted_labels)

def load_clip(path_to_model):
    model = CLIP(n_blocks, vocab_size)
    weights = torch.load(path_to_model)
    model.load_state_dict(weights)
    model.eval()
    model = torch.compile(model)
    return model

def extract_logits(model, reviews, images):
    temperature = model.temperature
    text_embeddings, image_embeddings = [], []
    for review, image in zip(reviews, images):
        text = tokenizer(review)
        image = iamge_transform(image)
        with torch.no_grad():   
            image_emb = model.project_image(image.unsqueeze(0))
            text_emb = model.project_text(text.unsqueeze(0))
            text_embeddings.append(text_emb)
            image_embeddings.append(image_emb)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    logits = (image_embeddings @ text_embeddings.t()) / torch.exp(temperature)
    return logits 


clip = load_clip(weights_path)

def iterate(model, n_iters, n_rest):
    accuracies = []

    for _ in range(n_iters):
        rest = sample_n_rest(n_rest)
        labels = rest.label.tolist()
        logits = extract_logits(model, rest.review.tolist(), [Image.open(image) for image in rest.image_path.tolist()]) 
        accuracy = calculate_accuracy(logits, labels)
        accuracies.append(accuracy)

    return np.mean(accuracies)

if __name__ == "__main__":
    n_iters = 1000
    n_rest = 3
    print('Iterating..')
    accuracy = iterate(clip, n_iters, n_rest)
    print(f"Accuracy at {n_rest}: {accuracy}")