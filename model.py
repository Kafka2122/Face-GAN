import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class CreateStyleVector(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=0, bias=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
 
        
    def in_depth(self, img2):
        out = self.layer1(img2)
        out = self.layer2(out)
        out = self.layer3(out)
        
        return out

    def shortcut(self, img2):
        out = self.conv1x1(img2)
        return out

    def forward(self, img2):
        x = self.shortcut(img2) 
        out = self.in_depth(img2)
        output = torch.cat((out,x),dim = 1)
        
        return output


import torch
import torch.nn as nn

class VGGDecoder(nn.Module):
    def __init__(self):
        super(VGGDecoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(1000, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1,output_padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,output_padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Assuming output should be in the range [0, 1]
        )

    def forward(self, x):
        return self.decoder_layers(x)


class Generator(nn.Module):
    def __init__(self, img_size=128, latent_dim=64):
        super(Generator, self).__init__()
        self.style_vector_creator = CreateStyleVector()
        self.decoder = VGGDecoder()
        self.fc1 = nn.Linear(32*32*32, 256)
        self.vgg16 = models.vgg16()
        self.norm1 = AdaIN(256,128)
        self.norm2 = AdaIN(256,256)
        self.norm3 = AdaIN(256,512)
        self.norm4 = AdaIN(256,512)
        self.norm5 = AdaIN(256,512)

        self.encode1 = nn.Sequential(*list(self.vgg16.features.children())[:6])

        self.encode2 = nn.Sequential(*list(self.vgg16.features.children())[6:12])

        self.encode3 = nn.Sequential(*list(self.vgg16.features.children())[12:18])

        self.encode4 = nn.Sequential(*list(self.vgg16.features.children())[18:24])

        self.encode5 = nn.Sequential(*list(self.vgg16.features.children())[24:])
        self.fc2 = nn.Linear(8192, 25088)
        self.vgg_classifier = nn.Sequential(*list(self.vgg16.classifier.children()))

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, 1, 1),  # Adjust input and output channels to match original input
            nn.Sigmoid()
        )

    
    def forward(self, img1, img2):
        style_img = self.style_vector_creator(img2)
        
        lat_vector = self.fc1(style_img.reshape(style_img.size(0),-1))
        
        # Encoding
        x = self.encode1(img1)
       
        x = self.norm1(x,lat_vector)
        
        x = self.encode2(x)
        x = self.norm2(x,lat_vector)
        x = self.encode3(x)
        x = self.norm3(x,lat_vector)
        x = self.encode4(x)
        x = self.norm4(x,lat_vector)
        x = self.encode5(x)
        x = self.norm5(x,lat_vector)
        x = x.reshape(x.size(0), -1)
        
        x = self.fc2(x)
        x = self.vgg_classifier(x)
        x = x.view(x.size(0), 1000, 1, 1)
        decoded_output = self.decoder(x)
        
        return decoded_output

class Discriminator(nn.Module):
    def __init__(self, img_size=128):  
        super(Discriminator, self).__init__()
        self.vgg16_backbone = models.vgg16()
        self.fc_layer1 = nn.Linear(1000,200)
        self.fc_layer2 = nn.Linear(200,20)
       
        self.fc_final = nn.Linear(20, 1)

    def forward(self, img):
        x = self.vgg16_backbone(img)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_final(x)
        return x

        