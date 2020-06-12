#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML

import itertools
import csv
import yaml
import sys
import math

from PIL import Image

from datetime import datetime
import os
import shutil

sys.path.append("./Modelli")

import GenDis_SA as gd2




# Adam parameters
#=================
# alpha. Also referred to as the learning rate or step size. 
#        The proportion that weights are updated (e.g. 0.001). 
#        Larger values (e.g. 0.3) results in faster initial learning 
#        before the rate is updated. Smaller values (e.g. 1.0E-5) 
#        slow learning right down during training

# beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).

# beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). 
#        This value should be set close to 1.0 on problems with a sparse gradient 
#        (e.g. NLP and computer vision problems).

# epsilon. Is a very small number to prevent any division by 
#          zero in the implementation (e.g. 10E-8).


def salvaCSV(lista, nomefile):
# salva le liste come file csv
    with open(nomefile, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lista)


def saveFakeImages(fake, nomefile):
    temp = vutils.make_grid(fake, padding=2, normalize=True)
    img2 = temp.numpy().transpose(1,2,0)

    # normalizzazione
    min_ = np.min(img2)
    max_ = np.max(img2)
    img2 = np.add(img2, -min_)
    img2 = np.divide(img2, (max_-min_ +1e-5))
    img2 = np.multiply(img2,255.0)
    img2 = np.uint8(img2)

    img2 = Image.fromarray(img2)
    img2.save(nomefile)



def salvaProvino(nomeDir, nomeFile, netG, netD, fixed_noise):
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    nomeFileImage = os.path.join(nomeDir, nomeFile +".jpg")
    saveFakeImages(fake, nomeFileImage)

    torch.save(netD, os.path.join(nomeDir, nomeFile +"__D.pth" ) )
    torch.save(netG, os.path.join(nomeDir, nomeFile +"__G.pth" ) )
    print("salvato il modello in ", nomeFile)


def main(pl, paramFile):

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Spatial size of training images. All images will be resized to this
    # size using a transformer.
    image_size = pl["image_size"]
    # Size of feature maps in generator
    ngf = pl["ngf"]
    # Size of feature maps in discriminator
    ndf = pl["ndf"]
    # number of layers
    k = int(math.log(image_size, 2)) - 3
    # k = pl["k"]

    # creazione della directory dell'esperimento ==================================
    today = datetime.now()
    nomeDir = "./" + pl["nomeModello"] + "_" +today.strftime('%Y_%m_%d_%H_%M')
    os.mkdir(nomeDir)
    # copio il file di parametri nella dir dell'esperimento
    newPath = shutil.copy(paramFile, nomeDir)
    #==============================================================================

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    trasf = transforms.Compose([   transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

    full_dataset = dset.ImageFolder(root=pl["dataroot"], transform=trasf)
    
    ############################################################################
    ######## DEBUG DATALOADER ##################################################
    if pl["n_samples"] > 0 :
        sottoinsieme = list(range(0,pl["n_samples"]))
        dataset = torch.utils.data.Subset(full_dataset, sottoinsieme)
    else:
        dataset = full_dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=pl["batch_size"],  
                 shuffle=True, num_workers=pl["workers"])
    ############################################################################

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and pl["ngpu"] > 0) else "cpu")

    #======================================
    # Create the generator
    netG = gd2.Generator(pl["ngpu"], pl["nz"], ngf, pl["nc"], k).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (pl["ngpu"] > 1):
        netG = nn.DataParallel(netG, list(range(pl["ngpu"])), k)

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(gd2.weights_init)

    #--------------------------------------
    # Create the Discriminator
    netD = gd2.Discriminator(pl["ngpu"], ndf, pl["nc"], k).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (pl["ngpu"] > 1):
        netD = nn.DataParallel(netD, list(range(pl["ngpu"])))

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netD.apply(gd2.weights_init)

    # Print the model ==================================
    nomeFile = os.path.join(nomeDir, pl["nomeModello"]+"_architettura.txt")
    stringa = str(netD) +"\n\n"+ str(netG) 
    with open(nomeFile, "w") as text_file:
        text_file.write(stringa)
    print(stringa)
    #===================================================

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    #fixed_noise = torch.randn(64, pl["nz"], 1, 1, device=device)
    fixed_noise = torch.randn(pl["batch_size"], pl["nz"], 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=pl["lrd"], betas=(pl["beta1"], pl["beta2"]))
    optimizerG = optim.Adam(netG.parameters(), lr=pl["lrg"], betas=(pl["beta1"], pl["beta2"]))

    # Lists to keep track of progress
    G_losses = []
    D_losses = []


    print("Inizio apprendimento, tutti i dati saranno salvati in "+ nomeDir)
    # For each epoch
    for epoch in range(pl["num_epochs"]):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
        
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, pl["nz"], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()


            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, pl["num_epochs"], 
                        i, len(dataloader),
                        # Loss_D    Loss_G
                        errD.item(), errG.item(), 
                        #D(x)   D(G(z))
                        D_x,    D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        



        # salva ad ogni epoca un provino delle immagini generate
        # ed i modelli relativi
        nomeFile = pl["nomeModello"]+ "_" +str(epoch)
        salvaProvino(nomeDir, nomeFile, netG, netD, fixed_noise)

    #=======================================
    # Salvataggio di fine training
    nomeFile = pl["nomeModello"]+ "_"+"FINALE"
    salvaProvino(nomeDir, nomeFile, netG, netD, fixed_noise)



    nomeFile = os.path.join(nomeDir, pl["nomeModello"] + "_"+pl["nomeFileLosses"][0])
    salvaCSV(G_losses, nomeFile)
    nomeFile = os.path.join(nomeDir, pl["nomeModello"] + "_"+pl["nomeFileLosses"][1])
    salvaCSV(D_losses, nomeFile)
    print("salvati i dati dell'apprendimento in ", nomeDir)




if __name__ == "__main__":
    inputFile = sys.argv[1]

    # Lettura parametri dal file yaml
    with open(inputFile) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_list = yaml.load(file, Loader=yaml.FullLoader)

    main(param_list, inputFile)
