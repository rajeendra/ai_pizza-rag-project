import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. CONFIGURATION ---
BLOCK_SIZE = 8       # Max context length for prediction
N_EMBED = 32         # Embedding dimension
N_HEAD = 4           # Number of attention heads
N_LAYER = 1          # Number of transformer blocks
# VOCAB_SIZE is now a placeholder, the actual size is calculated in training
DEFAULT_VOCAB_SIZE = 65 
DEVICE = 'cpu'

# --- 2. THE CORE COMPONENTS (Self-Attention) ---
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )
    def forward(self, x):
        return self.net(x)

# --- 3. THE TRANSFORMER BLOCK ---
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 4. THE FULL MINIMAL LLM (FIXED: accepts vocab_size) ---
class MiniTransformer(nn.Module):
    """ The full model combining all parts """
    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE):
        super().__init__()
        # Token and Positional Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        
        # Transformer Blocks
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        
        # Final Layer
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = token_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    # --- Generation Function (Inference) ---
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- 5. DATA AND TRAINING ---
def get_batch(data, batch_size):
    """ Simple data loader to get inputs (x) and targets (y) """
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def train_and_save_model(model_path="mini_transformer.pth"):
    # Simple training data (must be converted to integer tokens)
    text = "hello world! this is a simple test sequence for a minimal transformer model."
    chars = sorted(list(set(text)))
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    
    # Calculate the actual vocabulary size
    actual_vocab_size = len(chars)
    
    # Encoder: string to list of integers
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    
    # Train/Val split
    data = encode(text)
    n = int(0.9*len(data))
    train_data = data[:n]
    
    # Initialize model with the actual calculated size (FIX)
    model = MiniTransformer(vocab_size=actual_vocab_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 
    #
    # Training a 1-layer, 32-dim Transformer...
    #
    # Step 0: Loss 2.9755
    # Step 100: Loss 1.7028
    # Step 200: Loss 1.2000
    # Step 300: Loss 0.6226
    # Step 400: Loss 0.5471
    # Step 500: Loss 0.3328
    # Step 600: Loss 0.3647
    # Step 700: Loss 0.2180
    # Step 800: Loss 0.2427
    # Step 900: Loss 0.2381
    #
    # Model saved to mini_transformer.pth
    
    print(f"Training a {N_LAYER}-layer, {N_EMBED}-dim Transformer with VOCAB_SIZE={actual_vocab_size}...")
    
    # Training Loop
    BATCH_SIZE = 4
    for iter in range(1000):
        xb, yb = get_batch(train_data, BATCH_SIZE)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")

    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # print(f"\nmodel_state_dict: {model.state_dict()}")
    # print(f"\stoi: {stoi}")
    # print(f"\nitos: {itos}")

    # model_state_dict: OrderedDict([('token_embedding_table.weight', tensor([[-0.1513,  1.8141, -1.7976, -1.8154, -1.9491, -0.3378, -1.3129, -0.5292,
    #          -0.1371,  0.6082,  1.4283, -1.0863,  0.6770,  0.4591, -1.3269,  0.5926,
    #           0.2122, -1.7282, -0.5751, -1.8664,  1.2250, -1.1070, -0.3705, -0.2650,
    #          -0.8428, -1.6264, -1.1180, -0.5637, -0.2983, -0.5125,  0.7404, -1.2482],
    #         [ 0.5440, -0.3277, -0.4345, -0.5152,  0.5244, -0.3857,  1.6006, -0.1771,
    #           0.2106, -0.5896,  1.0429,  2.6498, -0.3277, -0.0559,  0.5836, -0.1657,
    #          -0.5843, -2.4662, -0.3860, -0.5831, -1.7011, -0.5006,  0.8093,  0.6821,
    #           0.2270,  0.3632, -1.1440,  0.4735, -0.6224, -0.2008, -0.4965,  1.1228]])), , ('blocks.0.sa.heads.0.tril', tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1.]])), ('blocks.0.sa.heads.0.key.weight', tensor([[ 0.2885, -0.0204, -0.0848,  0.0273,  0.0961,  0.1076, -0.1008, -0.0833,
    #          -0.0258, -0.1010, -0.2674,  0.0532, -0.2045, -0.3047,  0.0103, -0.3064,
    #           0.0386,  0.1730, -0.2088, -0.0131,  0.1244,  0.0870,  0.1911,  0.0121,
    #         -0.0305,  0.2014,  0.0781, -0.0046,  0.1023, -0.0612,  0.2036, -0.0090,
    #          0.1160, -0.0505, -0.0386, -0.1073, -0.0178, -0.1039,  0.2111, -0.0774])), ('blocks.0.ffwd.net.2.weight', tensor([[ 1.6586e-01,  3.6004e-02, -1.9876e-01,  ...,  2.0962e-01,
    #           5.4728e-02,  1.3762e-03],
    #         [ 1.4565e-01,  5.5355e-02, -2.0392e-02,  ...,  5.5507e-02,
    #          -8.2383e-02,  1.6303e-01],
    #         [-9.6314e-02,  8.4538e-02, -7.7841e-03,  ...,  5.9602e-04,
    #           8.7924e-02,  1.3121e-01],
    #         ...,
    #         [-8.3415e-02, -1.0594e-01, -1.9513e-02,  ..., -7.5940e-02,
    #          -1.1837e-04, -2.6821e-01],
    #         [ 2.1560e-02, -9.0726e-03,  8.4098e-02,  ..., -1.2726e-01,
    #          -5.5282e-03, -6.3274e-02],
    #         [ 9.6738e-03,  9.0655e-02,  4.1928e-02,  ..., -1.9553e-01,
    #          -8.4879e-03, -5.8017e-02]])) , ('lm_head.weight', tensor([[-0.2393,  0.1618,  0.1359,  0.1302, -0.0767, -0.2420, -0.0789, -0.3507,
    #           0.2766, -0.3658,  0.2084,  0.0822, -0.1243, -0.1144,  0.1113, -0.3092,
    #           0.2452, -0.1628,  0.2771, -0.4119,  0.2086, -0.0171, -0.2081,  0.1149,
    #          -0.1249,  0.0467,  0.3641, -0.0417,  0.2891, -0.1586,  0.2887,  0.2895],
    #         [ 0.2494,  0.1607, -0.1947, -0.1977, -0.1093,  0.0634, -0.1293,  0.1441,
    #          -0.1910, -0.0610,  0.0889,  0.3199, -0.0354, -0.2957,  0.0597,  0.2888,
    #          -0.2196, -0.2931, -0.0940, -0.1341,  0.0579,  0.2080,  0.2594,  0.2659,
    #           0.0057,  0.1213, -0.2265, -0.1162, -0.3711,  0.3330,  0.2732, -0.0525]])), ('lm_head.bias', tensor([ 0.0875, -0.0736, -0.1234,  0.0211, -0.2434, -0.0751, -0.1079, -0.0480,
    #         -0.0143,  0.0115,  0.0741, -0.1259, -0.0086, -0.0640, -0.0942, -0.1999,
    #          0.0686,  0.0147, -0.1487, -0.1004, -0.1581]))])

    # Model saved to: mini_transformer.pth
    # \stoi: {' ': 0, '!': 1, '.': 2, 'a': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'h': 8, 'i': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'w': 20}

    # itos: {0: ' ', 1: '!', 2: '.', 3: 'a', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'h', 9: 'i', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's', 18: 't', 19: 'u', 20: 'w'}
    # (venv) rajeendra@admins-mbp pizza-rag-project % 


if __name__ == '__main__':
    train_and_save_model()