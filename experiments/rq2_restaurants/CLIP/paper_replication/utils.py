import torch
import torch.nn as nn 
import torch.nn.functional as F
from nltk.corpus import words

def image_to_patches(image, patch_size=16):
    """
    Convert an image into patches of a specified size.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (int): Size of the patches. Default is 16.
        
    Returns:
        torch.Tensor: Tensor containing image patches of shape (N, C, patch_size, patch_size),
                      where N is the number of patches.
    """
    # Ensure input image has 3 dimensions (channels, height, width)
    if len(image.shape) != 3:
        raise ValueError("Input image must have shape (C, H, W)")
    
    # Extract dimensions
    B, C, H, W = image.shape
    
    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Reshape image into patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(num_patches, C, patch_size, patch_size)
    
    return patches.view(-1)


def image_to_patches_batch(images, patch_size=16):
    """
    Convert a batch of images into patches of a specified size.
    
    Args:
        images (torch.Tensor): Input image tensor of shape (B, C, H, W), where B is the batch size.
        patch_size (int): Size of the patches. Default is 16.
        
    Returns:
        torch.Tensor: Tensor containing image patches of shape (B, N, C, patch_size, patch_size),
                      where B is the batch size, and N is the number of patches.
    """
    # Ensure input images have 4 dimensions (batch size, channels, height, width)
    if len(images.shape) != 4:
        raise ValueError("Input images must have shape (B, C, H, W)")
    
    # Extract dimensions
    B, C, H, W = images.shape
    
    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Reshape images into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, num_patches, C, patch_size, patch_size)
    
    return patches.view(B, -1)


class ImageTokenizer(nn.Module):
    def __init__(self, patch_size=16, image_size=224, image_dim=768):  
        super(ImageTokenizer, self).__init__()
        self.patch_size = patch_size
        self.image_dim = image_dim
        self.n_patches = int(image_size*image_size / (patch_size**2)) 
        self.n_flattened_patches = int(self.n_patches * 3 * patch_size * patch_size)
        self.image_flattener = nn.Linear(self.n_flattened_patches, self.image_dim)
        self.ln = nn.LayerNorm(self.image_dim)

    def forward(self, image):
        image = image_to_patches_batch(image)
        image = self.ln(self.image_flattener(image))
        return image


def load_nltk_vocabulary():
    english_words = words.words()
    english_words = [word.lower() for word in english_words]
    english_words = list(set(english_words))
    vocab_size = len(english_words)
    return english_words, vocab_size

def load_norvig_vocabulary():
    with open('/home/agambetti/PhD/multi-media-interaction/experiments/rq2_restaurants/CLIP/paper_replication/data/vocabulary_30k.txt', 'r') as file:
        vocabulary = file.read().splitlines()
        vocabulary = [word.strip() for word in vocabulary]
    vocab_size = len(vocabulary)
    return vocabulary, vocab_size


class TextTokenizer(nn.Module):
    def __init__(self, dictionary, max_length=76):
        super(TextTokenizer, self).__init__()
        self.dictionary = dictionary
        self.vocab_size = len(dictionary)
        self.oov_token = self.vocab_size # out of vocabulary token
        self.pad_token = self.vocab_size + 1
        self.sos = "[SOS]"
        self.eos = "[EOS]"
        self.sos_id = self.vocab_size + 2
        self.eos_id = self.vocab_size + 3
        self.stoi = {word: i for i, word in enumerate(dictionary)}
        self.itos = {i: word for word, i in self.stoi.items()}
        self.max_length = max_length

    def encode(self, text):
        text = text.lower().strip()
        tokens = text.split()
        encoded = torch.tensor([self.stoi.get(token, self.oov_token) for token in tokens])
        encoded = torch.cat((torch.tensor([self.sos_id]), encoded))

        if len(encoded) < self.max_length:
            padding = torch.full((self.max_length - len(encoded),), self.pad_token, dtype=torch.long)
            encoded = torch.cat((encoded, padding))
        else:
            encoded = encoded[:self.max_length]
        
        encoded[-1] = self.eos_id
        return encoded 


    def decode(self, tokens):
        return ' '.join([self.itos.get(token, 'OOV') for token in tokens])
    
    def __call__(self, text):
        return self.encode(text)


class PatchEmbeddings(nn.Module):
    def __init__(self, embed, patch_size, in_channels=3):
        super(PatchEmbeddings, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed, patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c) # flatten 
        return x


class PositionalEmbeddings(nn.Module):
    def __init__(self, embed, max_len=10_000):
        super(PositionalEmbeddings, self).__init__()
        self.positional_embeddings = nn.Parameter(torch.zeros(max_len, 1, embed), requires_grad=True)

    def forward(self, x):
        pe = self.positional_embeddings[:x.shape[0]]
        out = x + pe
        return out


class ImageTokenizer2(nn.Module):
    def __init__(self, embed, patch_size):
        super(ImageTokenizer2, self).__init__()
        self.patch_embeddings = PatchEmbeddings(embed, patch_size)
        self.positional_embeddings = PositionalEmbeddings(embed)
        self.cls = nn.Parameter(torch.randn(1, 1, embed), requires_grad=True)
        self.ln = nn.LayerNorm(embed)
    
    def forward(self, x):
        x = self.patch_embeddings(x)
        x = self.positional_embeddings(x)
        cls = self.cls.expand(-1, x.shape[1], -1) # create cls token 
        x = torch.cat([cls, x]) # aggregate cls token 
        x = x.permute(1, 0, 2) # get back to batch size x patches x embed 
        return self.ln(x) # add layer norm as in CLIP 



if __name__ == '__main__':
    # Example usage:
    # Load an image
    image = torch.randn(16, 3, 224, 224)  # Example image with shape (3, 224, 224)
    #image_to_patches_batch(image)
    embed = 768
    patch_size = 16
    image_tokenizer2 = ImageTokenizer2(embed, patch_size)
    out = image_tokenizer2(image)
    #print(out.shape)


    vocab, vocab_size = load_norvig_vocabulary()
    #print(words[:50])

    tokenizer = TextTokenizer(vocab)
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(text)
    print(tokens.shape)
    #print(len(tokens))

    #print(vocab_size)
