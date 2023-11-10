import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import torch
from tqdm import tqdm
import numpy as np

def train_VAE(model, train_loader, test_loader, epochs, optimizer, model_path, device):
    best_test_loss = 999999999999

    # Training loop
    for epoch in range(1, epochs+1):
        model.to(device)
        model.train()  # Set model to train mode

        train_loss = 0.0
        for batch_data, _ in tqdm(train_loader, leave=False, desc='Epoch {}/{}'.format(epoch, epochs)):
            batch_data = batch_data.to(device)

            assert not torch.isnan(batch_data).any(), "Tensor contains NaN values"

            # Forward pass
            recon_data, mu, logvar = model(batch_data)

            # Compute loss
            loss = model.loss(recon_data, batch_data, mu, logvar)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_data) 

        # Compute average training loss for the epochreparameterize
        train_loss /= len(train_loader.dataset)

        # Evaluate on the test set
        model.eval()  # Set model to evaluation mode

        test_loss = 0.0
        with torch.no_grad():
            for batch_data, _ in test_loader:
                batch_data = batch_data.to(device)
                recon_data, mu, logvar = model(batch_data)
                loss = model.loss(recon_data, batch_data, mu, logvar)
                test_loss += loss.item() * len(batch_data)

        # print('test', test_loss)
        # Compute average test loss for the epoch
        test_loss /= len(test_loader.dataset)
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), f=model_path)
            best_test_loss = test_loss

        # Print training and test progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")
