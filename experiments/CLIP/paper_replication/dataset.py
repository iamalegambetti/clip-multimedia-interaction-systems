from torch.utils.data import Dataset
from PIL import Image 
import re, html, os, random
import pandas as pd

# fix a random seed 
random.seed(46)

def clean_html(text):
    text = str(text)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    cleantext = html.unescape(cleantext)
    return cleantext

class RestaurantDataset(Dataset):
    def __init__(self, reviews, images_path, tokenizer=None, image_transforms=None):
        super(RestaurantDataset).__init__()
        self.reviews = pd.read_csv(reviews)
        self.reviews['review'] = self.reviews['review'].apply(lambda x: clean_html(x))
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        restaurant = self.reviews.loc[idx, 'Restaurant']
        reviewer = self.reviews.loc[idx, 'Reviewer']
        review = self.reviews.loc[idx, 'review']
        images = [os.path.join(self.images_path, restaurant, f) for f in os.listdir(os.path.join(self.images_path, restaurant)) if reviewer in f]
        image = random.choices(images, k=1)
        image = Image.open(image[0])
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.tokenizer: 
            review = self.tokenizer(review)
        return {'image':image, 'review':review}


if __name__ == '__main__':
    from utils import TextTokenizer
    from torchvision import transforms

    reviews_path = "clip/data/test_reviews.csv"
    images_path = "/Volumes/Transcend/data/Yelp/NY/images"

    tokenizer = TextTokenizer()

    iamge_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels about the center
        transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    dataset = RestaurantDataset(reviews_path, images_path, tokenizer=tokenizer, image_transforms=iamge_transform)
    print(len(dataset))
    d1 = dataset[6]


    for i in range(len(dataset)):
        if i % 1000 == 0: print(i)
        try:
            d = dataset[i]
        except Exception as e:
            print(e)
            print(i)
 
    