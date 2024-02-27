import torch, random, os
from transformers import CLIPProcessor, CLIPModel
import pandas as pd 
from PIL import Image 
from sklearn.metrics import accuracy_score
random.seed(46)

fine_tuned = False
images_path = "/home/agambetti/data/Yelp/NY/images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16") # https://huggingface.co/openai/clip-vit-base-patch16
model = torch.compile(model)
model.to(device)

def list_images(restaurant, reviwer):
    images = os.listdir(os.path.join(images_path, restaurant))
    images = [image for image in images if reviwer in image]
    return images

if fine_tuned:
    path_to_model = "/home/agambetti/PhD/multi-media-interaction/experiments/CLIP/fine_tuned_embeddings/weights_ft/clip_1.pt"
    weights = torch.load(path_to_model)
    model.load_state_dict(weights)

df = pd.read_csv("/home/agambetti/PhD/multi-media-interaction/experiments/CLIP/paper_replication/data/test_reviews.csv")
df['images'] = df.apply(lambda x: list_images(x.Restaurant, x.Reviewer), axis=1)
df['image'] = df.apply(lambda x: random.choice(x.images) if len(x.images) > 0 else None, axis=1)
df['image_path'] = df.apply(lambda x: os.path.join(images_path, x.Restaurant, x.image) if x.image else None, axis=1)

def sample_observation(df, n=1):
    observation = df.sample(n)
    review = observation['review'].item()
    image = observation['image_path'].item()
    image = Image.open(image)
    return review, image
    
def create_pair(df, n_negatives=1):
    review1, image1 = sample_observation(df) # positives 
    negatives = [sample_observation(df)[0] for _ in range(n_negatives)] # negatives
    output = {'positive':review1, 'image':image1, 'negative':negatives}
    return output 

def align(data):
    positive, negative, image = data['positive'], data['negative'], data['image']
    inputs = processor(text = [positive] + negative, images=image, return_tensors="pt", padding=True, max_length=77, truncation=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    index = torch.argmax(probs).item()
    return index 


def main(n_negatives = 1, n_iters=1_000):

    predictions = []
    labels = [0] * n_iters

    for iter in range(n_iters):
        #if iter % 100 == 0: print(f'Processing iteration {iter}..')
        data = create_pair(df, n_negatives=n_negatives)
        try:
            pred = align(data)
            predictions.append(pred)
        except Exception as e:
            print(e)
            continue
    
    labels = labels[:len(predictions)]
    accuracy = accuracy_score(labels, predictions)
    length = len(predictions)
    print(f'Accuracy: {accuracy}, Length Predictions: {length}.')
    return accuracy



def test():
    data = create_pair(df, n_negatives=3)
    print(data)

if __name__ == '__main__':
    print('Evaluating..')
    main(n_negatives=10, n_iters=1_000)
    #test()

#outputs = model(**inputs)
#logits_per_image = outputs.logits_per_image # this is the image-text similarity score
#probs = logits_per_image.softmax(dim=1)
