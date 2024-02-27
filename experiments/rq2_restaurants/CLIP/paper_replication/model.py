import torch 
import torch.nn as nn 
from encoder import Encoder as ImageEncoder
from decoder import Decoder as TextEncoder
from utils import ImageTokenizer, ImageTokenizer2, PositionalEmbeddings

class Project(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Project, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        return self.ln(self.linear(x))

class CLIP(nn.Module):
    def __init__(self, n_blocks, vocab_size, embed=512, n_heads=12, head_dim=64, patch_size=16, image_size=224):
        super(CLIP, self).__init__()
        
        self.image_dim = 768
        self.text_dim = 512
        self.embed = embed
        self.image_size = image_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(self.vocab_size + 4, self.text_dim) # text embedding 
        self.positional_embeddings = PositionalEmbeddings(self.embed) # for text 

        self.image_encoder = ImageEncoder(n_blocks, n_heads, self.image_dim, head_dim)
        self.text_encoder = TextEncoder(n_blocks, n_heads, self.text_dim, head_dim)

        self.image_to_patches = ImageTokenizer2(self.image_dim, patch_size)
        self.image_projection = Project(self.image_dim, self.embed)
        self.text_projection = Project(self.text_dim, self.embed)

        self.temperature = nn.Parameter(torch.tensor(0.07), requires_grad=True)

    def forward(self, image, text):
        image = self.image_to_patches(image)
        image = self.image_encoder(image)
        image = image[:, 0, :] # take the [CLS] token
        image = self.image_projection(image)

        text = self.embedding(text)
        text = self.positional_embeddings(text)
        text = self.text_encoder(text) 
        text = text[:, -1, :] # take the [EOS] token
        text = self.text_projection(text)
    
        logits = (image @ text.t()) / torch.exp(self.temperature)
        return logits

    def project_image(self, image):
        image = self.image_to_patches(image)
        image = self.image_encoder(image)
        image = image[:, 0, :]
        image = self.image_projection(image)
        return image
    
    def project_text(self, text):
        text = self.embedding(text)
        text = self.text_encoder(text) 
        text = text[:, -1, :]
        text = self.text_projection(text)
        return text 


if __name__ == '__main__':
    # test
    from utils import load_norvig_vocabulary
    from utils import TextTokenizer

    vocab, vocab_size = load_norvig_vocabulary()
    n_blocks = 12
    n_heads = 12
    embed_dim = 512
    head_dim = 64
    image_size = 224
    B = 32

    clip = CLIP(n_blocks, vocab_size=vocab_size)

    # count the number of parameters in clip
    print(f'The model has {sum(p.numel() for p in clip.parameters() if p.requires_grad):,} trainable parameters')

    # generate a random image tensor with random integers from 0 to 255
    image = torch.randint(0, 256, (B, 3, image_size, image_size)).float()
    text = "hello this is a text"
    tokenizer = TextTokenizer(vocab)
    tokens = tokenizer(text)
    # expand tokens to have size B x len(tokens)
    tokens = tokens.expand(B, -1)
    #print(tokens.shape)

    logits = clip(image, tokens)
    #print(logits.shape) 
    
    # extract all the logits row by row 


