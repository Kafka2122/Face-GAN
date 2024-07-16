This code is written to train the model on Digi1M-Face dataset. This dataset was given in Assignment 2 of the course. 
Download the dataset and run the train_2.ipynb notebook. 

Here we use the pretrained weights of the model trained on CelebHq-A dataset. If you dont have pretrained model please comment out these parts:

"gen_weights_path = '/DATA/shrestha/midsem_project/P1_dataset_training/model_saved/face_gan.pth'
disc_weights_path = '/DATA/shrestha/midsem_project/P1_dataset_training/model_saved/dicriminator.pth'
generator.load_state_dict(torch.load(gen_weights_path))
discriminator.load_state_dict(torch.load(disc_weights_path))"

