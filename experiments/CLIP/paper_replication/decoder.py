import torch 
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

class MaskedAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(MaskedAttentionHead, self).__init__()
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
        ### BEGING MASKING ###
        weights = torch.tril(weights) # set to 0s the upper diagonal of the matrix -> do not attend prior tokens 
        weights = torch.where(weights == 0, float('-inf'), weights) # set to -inf the 0s -> when taking the softmax they will be 0
        ### END MASKING ###
        weights = F.softmax(weights, dim = -1) 
        logits = weights @ v
        return logits

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(MaskedMultiHeadAttention, self).__init__()
        self.n_heads = n_heads 
        self.multi_head_attention = nn.ModuleList([MaskedAttentionHead(embed_dim, head_dim) for _ in range(n_heads)])
        self.project = nn.Linear(n_heads * head_dim, embed_dim)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.multi_head_attention], dim = -1)
        x = self.project(x)
        return x
    
class BlockDecoder(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(BlockDecoder, self).__init__()
        self.MaskedMHA = MaskedMultiHeadAttention(n_heads, embed_dim, head_dim)
        self.FF = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.ln1(x + self.MaskedMHA(x))
        x = self.ln2(x + self.FF(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_blocks, n_heads, embed_dim, head_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([BlockDecoder(n_heads, embed_dim, head_dim) for _ in range(n_blocks)])
        self.project = nn.Linear(n_blocks * embed_dim, embed_dim)

    def forward(self, x):
        out = torch.cat([d(x) for d in self.decoder], dim = -1)
        return self.project(out)


if __name__ == '__main__':
    
    n_blocks = 12
    n_heads = 12
    embed_dim = 768
    head_dim = 64
    decoder = Decoder(n_blocks, n_heads, embed_dim, head_dim)
    print(decoder)

    # count the number of parameters in decoder
    print(f'The model has {sum(p.numel() for p in decoder.parameters() if p.requires_grad):,} trainable parameters')

    x = torch.rand(10, embed_dim)
    out = decoder(x)
    print(out.shape) # torch.Size([1, 10, 512])
