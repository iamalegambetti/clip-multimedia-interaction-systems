import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.embed_dim = embed_dim # dimension of the embedding 
        self.head_dim = head_dim # dimension of the head 

        self.values = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.head_dim, bias=False)

    def forward(self, x):
        v = self.values(x)
        k = self.keys(x)
        q = self.queries(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** .5
        weights = F.softmax(weights, dim = -1) 
        logits = weights @ v
        return logits

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads # how many heads
        self.multi_head_attention = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(n_heads)])
        self.project = nn.Linear(n_heads * head_dim, embed_dim)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.multi_head_attention], dim = -1)
        x = self.project(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

class Block(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(Block, self).__init__()
        self.MHA = MultiHeadAttention(n_heads, embed_dim, head_dim)
        self.FF = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.ln1(x + self.MHA(x))
        x = self.ln2(x + self.FF(x))
        return x

class Encoder(nn.Module):
    def __init__(self, n_blocks, n_heads, embed_dim, head_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(*[Block(n_heads, embed_dim, head_dim) for _ in range(n_blocks)])
    
    def forward(self, x):
        x = self.encoder(x)
        return x

if __name__ == '__main__':
    # test
    n_blocks = 12
    n_heads = 12
    embed_dim = 512
    head_dim = 64
    x = torch.rand(10, embed_dim)
    encoder = Encoder(n_blocks, n_heads, embed_dim, head_dim)

    out = encoder(x)
    print(out.shape) 