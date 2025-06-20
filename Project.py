import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Reproducibility & device setup
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Preprocessing (with one-hot for categorical + printed split IPs)
def preprocess_df(df, sample_size=100_000):
    # 1) Sample
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    print("After sampling:", df.shape)

    # 2) Split IPs using provided method
    def split_ip(series, prefix):
        ip_split = series.str.split('.', expand=True)
        ip_split.columns = [f"{prefix}_{i+1}" for i in range(4)]
        ip_split = ip_split.astype(int)
        return ip_split

    orig_h = split_ip(df['id.orig_h'], 'orig_h')
    resp_h = split_ip(df['id.resp_h'], 'resp_h')

    # Print the split IPs
    print("\nSplit id.orig_h:")
    print(orig_h.head())

    print("\nSplit id.resp_h:")
    print(resp_h.head())

    # 3) Label encoder for labels
    label_le = LabelEncoder().fit(df['label'])

    # 4) One-hot encode proto, conn_state, history
    cat_cols = ['proto', 'conn_state', 'history']
    encoders = {}
    ohe_dfs = []
    for c in cat_cols:
        le = LabelEncoder().fit(df[c])
        encoders[c] = le
        idx = le.transform(df[c])
        ohe = pd.get_dummies(idx, prefix=c).astype(float)
        ohe_dfs.append(ohe)

    # 5) Normalize numeric features
    num_cols = ['id.orig_p', 'id.resp_p', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    scaler = StandardScaler().fit(df[num_cols])
    scaled = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols, dtype=float)

    # 6) Combine all features
    features = pd.concat([orig_h, resp_h] + ohe_dfs + [scaled], axis=1)
    features = features.astype(float)
    X = torch.tensor(features.values, dtype=torch.float32)
    print("Feature tensor shape:", X.shape)

    return X, scaler, encoders, num_cols, label_le, ohe_dfs

# Decode synthetic data back into readable format
def decode_synthetic(syn_np, scaler, encoders, num_cols, label_le, ohe_dfs):
    orig_h = np.floor(np.clip(syn_np[:, 0:4], 0, 255)).astype(int)
    resp_h = np.floor(np.clip(syn_np[:, 4:8], 0, 255)).astype(int)

    offs = 8
    decoded = {}
    for c, ohe in zip(encoders.keys(), ohe_dfs):
        length = ohe.shape[1]
        block = syn_np[:, offs:offs + length]
        idxs = np.argmax(block, axis=1)
        decoded[c] = encoders[c].inverse_transform(idxs)
        offs += length

    num_block = np.clip(syn_np[:, offs:], 0, None)
    num_real = scaler.inverse_transform(num_block)

    out = {
        'id.orig_h': ['.'.join(map(str, r)) for r in orig_h],
        'id.resp_h': ['.'.join(map(str, r)) for r in resp_h],
    }

    out.update(decoded)
    for i, col in enumerate(num_cols):
        out[col] = np.floor(num_real[:, i]).astype(int)

    return pd.DataFrame(out)

# 3. Load and preprocess data
df = pd.read_csv('C:/Users/dodo_/Downloads/CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

X, scaler, encoders, num_cols, label_le, ohe_dfs = preprocess_df(df)

# 4. Dataset and DataLoader
class NetDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]

dl = DataLoader(NetDataset(X), batch_size=256, shuffle=True)

# 5. GAN Models
noise_dim = 100
feature_dim = X.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, feature_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

gen = Generator().to(device)
disc = Discriminator().to(device)

# 6. Training loop
opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
crit = nn.BCELoss()
best_g = 1e9

for e in range(50):
    gl, dloss = [], []
    for real in dl:
        real = real.to(device)
        bs = real.size(0)
        valid = torch.ones(bs, 1, device=device)
        fake = torch.zeros(bs, 1, device=device)

        # Discriminator step
        opt_d.zero_grad()
        pred_r = disc(real)
        loss_r = crit(pred_r, valid)
        z = torch.randn(bs, noise_dim, device=device)
        pred_f = disc(gen(z).detach())
        loss_f = crit(pred_f, fake)
        (loss_r + loss_f).backward()
        opt_d.step()

        # Generator step
        opt_g.zero_grad()
        pred_g = disc(gen(torch.randn(bs, noise_dim, device=device)))
        loss_g = crit(pred_g, valid)
        loss_g.backward()
        opt_g.step()

        gl.append(loss_g.item())
        dloss.append(((loss_r + loss_f) / 2).item())

    ag, ad = np.mean(gl), np.mean(dloss)
    print(f"Epoch {e+1:02d} | D {ad:.3f} | G {ag:.3f}")
    if ag < best_g:
        best_g = ag
        torch.save(gen.state_dict(), 'generator.pt')

print("Done – best G loss:", best_g)

# 7. Generate synthetic traffic
gen.load_state_dict(torch.load('generator.pt', map_location=device))
gen.eval()
need = 20
with torch.no_grad():
    z = torch.randn(need, noise_dim, device=device)
    raw = gen(z).cpu().numpy()

# 8. Decode synthetic data
syn_df = decode_synthetic(raw, scaler, encoders, num_cols, label_le, ohe_dfs)
syn_df['label'] = ['Benign']*10 + ['Malicious']*10

# 9. Save and display
syn_df.to_csv('synthetic_traffic.csv', index=False)
print("Saved synthetic_traffic.csv:")
print(syn_df.head(10))

