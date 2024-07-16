import torch
from loss_fn import reconstruction_loss, diversity_loss, real_loss, fake_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_gen(gen_model, disc_model, input_img, optimizer):
    
    optimizer.zero_grad()
    gen_model.train()
    img1, img2 = input_img["img1"].permute(0, 3, 1, 2).float().to(device), input_img["img2"].permute(0, 3, 1, 2).float().to(device)
    
    gen_output = gen_model(img1, img2) # output by generator

    gen_cycle_output = gen_model(img1, img1)

    disc_output = disc_model(gen_output) # fake image being sent to discriminator 
    
    gen_rec_loss = reconstruction_loss(img1,gen_output) # recontruction_loss

    gen_cyc_loss = reconstruction_loss(img1, gen_cycle_output)

    d_loss_fake = real_loss(disc_output) # adversarial loss 
    

    div_loss = diversity_loss(img2, gen_output) # diversity loss we want genrated image to be far from the 2nd image
    
    gen_loss = gen_rec_loss  + d_loss_fake - div_loss + gen_cyc_loss

    gen_loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

    return gen_rec_loss, d_loss_fake*100, gen_cyc_loss, div_loss*1e-1,  gen_output

def train_disc(gen_model, disc_model, input_img, optimizer):

    
    disc_model.train()
    # train discriminator with fake images 
    img1, img2 = input_img["img1"].permute(0, 3, 1, 2).float().to(device), input_img["img2"].permute(0, 3, 1, 2).float().to(device)
    

    gen_output = gen_model(img1, img2)

    disc_fake_output = disc_model(gen_output) # fake image being sent to discriminator

    d_loss_fake = fake_loss(disc_fake_output)
    
    # train image with real image

    disc_real_output = disc_model(img1) # sending real image to dicriminator
    d_loss_real = real_loss(disc_real_output)

    total_loss = d_loss_fake + d_loss_real
    total_loss = total_loss*1e-4
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    return total_loss