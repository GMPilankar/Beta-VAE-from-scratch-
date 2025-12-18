# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 19:04:18 2025

@author: gaura
"""

from beta_vae import Beta_VAE
from torch.utils.data import DataLoader
from dataset import Dsprites_Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image#, ImageDraw, ImageFont
import os

def visualize_reconstruction(model, dataloader, num_samples=10):
    """
    Visualizes original vs reconstructed images on a large canvas.
    Top Row: Original
    Bottom Row: Reconstruction
    """
    model.eval()
    
    # 1. Get a single batch of data
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    # Ensure we don't ask for more samples than the batch has
    num_samples = min(num_samples, images.size(0))
    
    # 2. Pass through Model
    with torch.no_grad():
        mu, log_var, recon_images = model(images)
    
    # 3. Create the canvas
    # Height = 2 rows * 64 pixels
    # Width = num_samples columns * 64 pixels
    canvas_height = 64 * 2
    canvas_width = 64 * num_samples
    canvas = np.zeros((canvas_height, canvas_width))
    
    # 4. Paste images into canvas
    for i in range(num_samples):
        # --- Process Original ---
        orig = images[i].cpu().squeeze().numpy()
        
        # --- Process Reconstructed ---
        recon = recon_images[i].cpu().squeeze().numpy()
        
        # --- Paste into Grid ---
        # Row 0: Original
        canvas[0:64, i*64 : (i+1)*64] = orig
        
        # Row 1: Reconstruction
        canvas[64:128, i*64 : (i+1)*64] = recon

    # 5. Plotting
    plt.figure(figsize=(15, 4)) # Wide aspect ratio
    plt.imshow(canvas, cmap='Greys_r')
    
    # Add labels to the side (optional hack using text)
    plt.text(-30, 32, "Original", fontsize=12, rotation=90, va='center')
    plt.text(-30, 96, "Reconstructed", fontsize=12, rotation=90, va='center')
    
    plt.axis('off')
    plt.title(f"Reconstruction Quality (First {num_samples} samples)")
    plt.tight_layout()
    plt.show()


def create_traversal_gif(model, input_image, target_dim, filename="traversal.gif"):
    """
    Creates a GIF varying a single specific dimension.
    """
    model.eval()
    
    # 1. Get the Anchor Z (Mean of the posterior)
    with torch.no_grad():
        mu, log_var = model.encoder(input_image.unsqueeze(0))
    
    sd = torch.exp(0.5 * log_var).squeeze(0)
    #print("sd: ", sd)
    
    base_z = mu[0] # [10]
    
    # 2. Define the motion (Ping-Pong effect: -3 -> +3 -> -3)
    # 30 frames for smoothness
    steps = torch.linspace(-3.0, 3.0, 20) 
    # Add reverse to make it loop smoothly
    steps = torch.cat([steps, steps.flip(0)]) 
    
    frames = []
    
    for val in steps:
        # Create the vector
        z = base_z.clone()
        z[target_dim] = val # Overwrite the target dimension
        
        # Decode
        with torch.no_grad():
            # Add batch dim for decoder: [1, 10]
            recon = model.decoder(z.unsqueeze(0))
        
        # Post-process: Tensor -> Numpy Image (0-255)
        
        img_data = recon.squeeze().cpu().numpy()
        
        
        # Normalize to 0-255 uint8
        img_data = (img_data * 255).astype(np.uint8)
        
        # Convert to PIL Image to draw text on it
        pil_img = Image.fromarray(img_data, mode='L') # 'L' = Grayscale
        pil_img = pil_img.resize((256, 256), Image.NEAREST) # Upscale for visibility
        
        # Optional: Add text label
        #draw = ImageDraw.Draw(pil_img)
        # You might need to load a font, or use default
        # draw.text((10, 10), f"Dim: {target_dim}\nVal: {val:.2f}", fill=255)
        
        frames.append(pil_img)
        
    # 3. Save as GIF
    # duration is milliseconds per frame. 50ms = 20fps
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=50,
        loop=0 # 0 means infinite loop
    )
    print(f"Saved {filename}")

def visualize_traversal(model, input_image, z_dim=10, traversal_range=3):
    """
    Traverses latent dimensions one by one to visualize what they control.
    """
    model.eval()
    # 1. Get the "Anchor" Z for this specific image
    # We need to add a batch dimension: [1, 64, 64] -> [1, 1, 64, 64]
    with torch.no_grad():
        mu, log_var = model.encoder(input_image.unsqueeze(0))
    
    sd = torch.exp(0.5 * log_var).squeeze(0)
    z = mu + (torch.exp(log_var * 0.5) * torch.randn_like(log_var))
    z = z[0]
    
    #z = mu[0]
    
    # 1. Define the traversal range (e.g., -3 to +3 sigma)
    # We create 11 steps: -3.0, -2.4, ... 0 ... +2.4, +3.0
    #interpolation = torch.linspace(-traversal_range, traversal_range, 11).to(device)
    
    # 2. Create a canvas to hold the images
    # 11 columns (steps), z_dim rows (dimensions)
    canvas = np.zeros((64 * z_dim, 64 * 11))
    
    # 3. Loop through each latent dimension
    for row in range(z_dim):
        sigma = sd[row]
        #interpolation = torch.linspace(mu[0,row] - (4*sigma), mu[0,row] + (4*sigma), 11).to(device)
        interpolation = torch.linspace(mu[0,row] - 3, mu[0,row] + 3, 11).to(device)
        # Create a base vector of zeros
        # Shape: [11, z_dim] -> Batch of 11 vectors
        #z = torch.zeros((11, z_dim)).to(device)
        z_batch = z.clone().repeat(11, 1)
        
        # Traverse the current dimension 'row'
        # We replace the column 'row' with our interpolation values
        
        
        z_batch[:, row] = interpolation
        
        # 4. Decode
        with torch.no_grad():
            # recon: [11, 1, 64, 64]
            recon = model.decoder(z_batch)
            
        # 5. Place in canvas
        for col in range(11):
            img = recon[col].cpu().squeeze().numpy()
            # Paste into the correct grid location
            canvas[row*64 : (row+1)*64, col*64 : (col+1)*64] = img

    # 6. Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='Greys_r')
    plt.axis('off')
    plt.title("Latent Space Traversal (Rows=Dims, Cols=Values)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model =  Beta_VAE(z_dim = 10)
    model.to(device)
    checkpoint = torch.load(r'checkpoints\current_model.pt', map_location=device)
    model_state_dict = checkpoint['model_state']
    model.load_state_dict(model_state_dict)
    
    directory = r"D:\work\projects\beta_VAE_from_scratch"  
    test_data = Dsprites_Dataset(directory, split = 'test')
    test_loader = DataLoader(test_data , batch_size = 128 , shuffle = False)
    
    img , _ = test_data[40]
    img = img.to(device)
    #mu , log_var = model.encoder(img.unsqueeze(0))
    
    
    # Run it
    # create gifs for latent traversal
    # for i in range(10):
    #     filename = f"traversal_latent_dim_{i}.gif"
    #     filename = os.path.join("gif", '4', filename)
    #     create_traversal_gif(model, img, target_dim=i, filename=filename)
    
    # create traversal canvas
    #visualize_traversal(model, img)
    
    # create reconstruction canvas
    visualize_reconstruction(model, test_loader, num_samples=10)
    

