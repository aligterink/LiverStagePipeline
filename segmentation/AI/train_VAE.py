import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

torch.set_default_dtype(torch.float64)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("Using non-cuda device: {}".format(device))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = F.relu(self.fc2(z))
        recon_x = torch.sigmoid(self.fc3(h))
        return recon_x

    def forward(self, x):
        # print(self.fc1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar




# Define loss function
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss
    loss = recon_loss + kl_loss
    return loss




class CellDataset(Dataset):
    def __init__(self, df, non_feature_columns=None):
        self.non_feature_columns = non_feature_columns
        self.meta_data = df[non_feature_columns] if non_feature_columns else None
        self.features = df.loc[:, ~df.columns.isin(non_feature_columns)] if non_feature_columns else df
        
        self.features = (self.features-self.features.mean())/self.features.std()

    def __len__(self):
        return len(self.features.index)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[[idx]].values, dtype=torch.float64), {k:list(v.values())[0] for k,v in self.meta_data.iloc[[idx]].to_dict().items()}
    
    def get_number_of_features(self):
        return len(self.features.columns)
    
df = pd.read_csv("/mnt/DATA1/anton/pipeline_files/results/features/lowres_dataset_selection_features.csv")
df = df.interpolate()

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

non_feature_columns = ['file', 'label', 'day', 'strain']
train_ds = CellDataset(train_df, non_feature_columns=non_feature_columns)
test_ds = CellDataset(test_df, non_feature_columns=non_feature_columns)


# Example usage
input_dim = train_ds.get_number_of_features()
hidden_dim = 5
latent_dim = 3
learning_rate = 0.001
batch_size = 32
epochs = 10

# Create VAE model
model = VAE(input_dim, hidden_dim, latent_dim)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)



# Training loop
for epoch in range(epochs):
    model.train()  # Set model to train mode

    train_loss = 0.0
    for batch_data, _ in train_loader:

        # Forward pass
        recon_data, mu, logvar = model(batch_data)

        # Compute loss
        loss = loss_function(recon_data, batch_data, mu, logvar)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(batch_data)

    # Compute average training loss for the epoch
    train_loss /= train_ds.__len__()

    # Evaluate on the test set
    model.eval()  # Set model to evaluation mode

    test_loss = 0.0
    with torch.no_grad():
        for batch_data, _ in test_loader:
            recon_data, mu, logvar = model(batch_data)
            loss = loss_function(recon_data, batch_data, mu, logvar)
            test_loss += loss.item() * len(batch_data)

    # Compute average test loss for the epoch
    test_loss /= test_ds.__len__()

    # Print training and test progress
    if epoch % 1 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")







encoded_samples = []
for sample, meta_data in test_ds:
    
    label = sample[1]
    # Encode image
    model.eval()
    with torch.no_grad():
        encoded_img  = model.encoder(sample)
    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)
    
encoded_samples = pd.DataFrame(encoded_samples)




px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1', color=encoded_samples.label.astype(str), opacity=0.7)