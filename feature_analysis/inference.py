import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import torch
import pandas as pd

def get_latent_embedding(model, loader, device, embeddings_path=None):
    encoded_samples = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            for sample in batch:
                sample = sample.to(device)
                mu, _ = model.encode(sample)
                mu = mu.flatten().cpu().numpy()

                encoded_sample = {f"Latent dimension {i}": enc for i, enc in enumerate(mu)}
                encoded_samples.append(encoded_sample)

    df = pd.DataFrame(encoded_samples)

    if embeddings_path:
        df.to_csv(embeddings_path, index=False)
    return df
