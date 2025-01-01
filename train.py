import argparse
from dataloader import Custom_Dataloader
from model import Generator, Discriminator
from utils import train_gen, train_disc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the GAN model")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='output',
        help="Path to save output images"
    )
    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()
    path = args.data_path
    output_path = args.output_path

    # Load datasets
    train_dataset = Custom_Dataloader(path, mode="train")
    test_dataset = Custom_Dataloader(path, mode="test")

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Define hyperparameters
    EPOCHS = 400
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-6)

    # Define visualization function
    def visualize_output(output, base_output_folder=output_path):
        trial_number = 1
        trial_folder = f'{base_output_folder}/trial_{trial_number}'
        if not os.path.exists(trial_folder):
            os.makedirs(trial_folder, exist_ok=True)
        else:
            while os.path.exists(f'{base_output_folder}/trial_{trial_number}'):
                trial_number += 1
            trial_folder = f'{base_output_folder}/trial_{trial_number}'
            os.makedirs(trial_folder, exist_ok=True)

        batch_size = output.shape[0]
        selected_indices = torch.randperm(batch_size)[:8]
        selected_samples = output[selected_indices]
        selected_samples = selected_samples.cpu().detach().numpy()
        selected_images = selected_samples.transpose(0, 2, 3, 1)

        for i in range(len(selected_images)):
            image = (selected_images[i] * 255).astype(np.uint8)
            plt.imsave(f'{trial_folder}/image_{i+1}.png', image)

    # Define paths to save the model
    generator_path = 'generator_model.pth'
    discriminator_path = 'discriminator_model.pth'

    for epoch in range(EPOCHS):
        tot_gen_loss = 0
        tot_discriminator_loss = 0
        tot_div_loss = 0
        tot_dis_loss = 0
        tot_cyc_loss = 0
        print(f"EPOCH: {epoch+1}")
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for data in tepoch:
                gen_recloss, discriminator_loss, cycle_loss, diversity_loss, output = train_gen(
                    gen_model=generator, 
                    disc_model=discriminator, 
                    input_img=data, optimizer=gen_optimizer
                )

                dis_loss = train_disc(
                    gen_model=generator, 
                    disc_model=discriminator,
                    input_img=data,
                    optimizer=disc_optimizer
                )
                
                tot_gen_loss += gen_recloss
                tot_discriminator_loss += discriminator_loss
                tot_cyc_loss += cycle_loss
                tot_div_loss += diversity_loss
                tot_dis_loss += dis_loss
            
            print(f"Reconstruction loss: {tot_gen_loss/len(train_dataloader)}, "
                  f"Cycle_loss: {tot_cyc_loss/len(train_dataloader)}, "
                  f"Diversity_loss: {tot_div_loss/len(train_dataloader)}, "
                  f"Adversarial_Generator_loss: {tot_discriminator_loss/len(train_dataloader)}, "
                  f"Adversarial_Discriminator_loss: {tot_dis_loss}")
            
            if epoch % 10 == 0:
                visualize_output(output)

            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator.state_dict(), discriminator_path)

# Entry point
if __name__ == "__main__":
    main()

