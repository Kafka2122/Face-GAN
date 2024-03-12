from dataloader import Custom_Dataloader
from model import Generator, create_style_vector, Discriminator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

path = "/DATA/shrestha/midsem_project/data"

train_dataset = Custom_Dataloader(path, mode="train")
test_dataset = Custom_Dataloader(path, mode="test")

batch_size =32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# defining model
generator = Generator()
discriminator = Discriminator()


# Defing hyperparameters
EPOCHS = 40
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
rec_loss = reconstruction_loss() 
div_loss = diversity_loss()
fk_loss = fake_loss()
rl_loss = real_loss()

for epoch in range(EPOCHS):
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data in tepoch:

            img1, img2 = data["img1"], data["img2"]
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train discriminator with real image
            discriminator.zero_grad()
            real_output = discriminator(img1)

            d_loss_real = rl_loss(real_output, real_labels)
            d_loss_real.backward()

            # train discriminator with fake image
            gen_img = generator(img1, img2)
            fake_output = discriminator(gen_img)
            
            d_rec_loss = rec_loss(gen_img, img1) # reconstruction loss
            d_loss_fake = fk_loss(fake_labels,fake_output)
            d_loss_fake.backward()
            d_rec_loss.backward()
            # train generator with respect to dicriminator output
            
