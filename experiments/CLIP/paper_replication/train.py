import torch 
import torch.nn as nn
from model import CLIP
from utils import TextTokenizer, load_nltk_vocabulary, load_norvig_vocabulary
from torchvision import transforms
from dataset import RestaurantDataset
from torch.utils.data import DataLoader
import logging, os 
import numpy as np 

# CONFIG
BATCH_SIZE = 16 # 32 converges faster and looks more optimal than
LR = .0001 
EPOCHS = 10
n_blocks = 6 
save_weights = True
enable_batch_logs = True
train_path = "multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/data/train_reviews.csv"
test_path = "multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/data/test_reviews.csv"
images_path = "/home/agambetti/data/Yelp/NY/images"
logs_file = 'multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/logs/logs.txt'
weights_dir = 'multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/weights'
logging.basicConfig(filename=logs_file, level=logging.INFO, filemode='a+')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Optimizing on device:', device)

# load tokenizers and image transforms
vocabulary, vocab_size = load_norvig_vocabulary()
tokenizer = TextTokenizer(vocabulary)
iamge_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# load dataset 
train_dataset = RestaurantDataset(train_path, images_path, tokenizer=tokenizer, image_transforms=iamge_transform)
test_dataset = RestaurantDataset(test_path, images_path, tokenizer=tokenizer, image_transforms=iamge_transform)
print('Dataset length:', len(train_dataset), len(test_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load model 
clip = CLIP(n_blocks, vocab_size)
clip = clip.to(device)
print(f'The model has {sum(p.numel() for p in clip.parameters() if p.requires_grad):,} trainable parameters')

# set optmizizer and loss function
criterion_image = nn.CrossEntropyLoss()
criterion_text = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(clip.parameters(), lr=LR)

# training loop
for epoch in range(1, EPOCHS):
    print('Epoch:', epoch)
    train_losses = []
    clip.train()
    for i, batch in enumerate(train_dataloader):
        if i % 100 == 0: 
            print('Batch:', i, 'Loss:', np.mean(train_losses) if train_losses else None)
            if enable_batch_logs:
                logging.info(f'Epoch: {epoch}, Batch: {i}, Loss: {np.mean(train_losses) if train_losses else None}')

        optimizer.zero_grad()

        image = batch['image'].to(device)
        text = batch['review'].to(device)
        labels = torch.arange(image.size(0)).to(device) # this is equal to the batch size 
        logits = clip(image, text)

        loss_image = criterion_image(logits.transpose(0, 1), labels) # axis 0 
        loss_text = criterion_text(logits, labels) # axis 1 
        loss = (loss_image + loss_text) / 2
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    average_train_loss = np.mean(train_losses)
    print('Average train loss:', average_train_loss)
    logging.info(f'Epoch: {epoch}, Average train loss: {average_train_loss}')

    print('Evaluating..')
    with torch.no_grad():
        losses = []
        clip.eval()
        for i, batch in enumerate(test_dataloader):
            if i % 100 == 0: print('Batch Test:', i)

            image = batch['image'].to(device)
            text = batch['review'].to(device)
            labels = torch.arange(image.size(0)).to(device)
            logits = clip(image, text)

            loss_image = criterion_image(logits.transpose(0, 1), labels)
            loss_text = criterion_text(logits, labels)
            loss = (loss_image + loss_text) / 2
            loss = loss.item()
            losses.append(loss)
        
        average_loss = np.mean(losses)
        print('Average loss:', average_loss)
        logging.info(f'Epoch: {epoch}, Average val loss: {average_loss}\n')

        # save weights
        if save_weights:
            torch.save(clip.state_dict(), os.path.join(weights_dir, f'clip_{epoch}.pt'))
            print('Weights saved')

    print()
