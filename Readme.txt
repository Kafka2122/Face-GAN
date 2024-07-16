This project is inspired by the "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation"
https://arxiv.org/abs/1711.09020

In this work I want to transfer features such as expression, pose, lighting from the secondary image into the primary image while maintaining the originality of the second image.

To train the model download the CelebHq-A dataset 
To download dataset run following command: "bash download.sh celeba-hq-dataset"

To train the model run "python train.py" the training output images will be stored to an "output" folder created 