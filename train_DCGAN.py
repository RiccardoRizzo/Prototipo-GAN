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

#-------------------------------------------------
def salvaCSV(lista, nomefile):
# salva le liste come file csv
    with open(nomefile, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lista)

#-------------------------------------------------
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


#---------------------------------------------------
def salvaCheckpoint(nomeDir, nomeFile, netD, netG, optimizerD, optimizerG, fixed_noise):
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    nomeFileImage = os.path.join(nomeDir, nomeFile +".jpg")
    saveFakeImages(fake, nomeFileImage)

    nomeCheckpoint = os.path.join(nomeDir, nomeFile +"__D.pth" )
    torch.save({
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            }, nomeCheckpoint)

    nomeCheckpoint = os.path.join(nomeDir, nomeFile +"__G.pth" )
    torch.save({
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'fixed_noise' : fixed_noise,
            }, nomeCheckpoint)
    print("salvato il modello in ", nomeFile)


#---------------------------------------------------------
def stringaStato(epoch, num_epochs, i, dataloader, errD, errG, D_x, D_G_z1, D_G_z2):
    out = ('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, 
                        i, len(dataloader),
                        # Loss_D    Loss_G
                        errD.item(), errG.item(), 
                        #D(x)   D(G(z))
                        D_x,    D_G_z1, D_G_z2)
          )
    return out


def creaG(ngpu, nz, ngf, nc, k, device):
    # Create the generator
    netG = gd2.Generator(ngpu, nz, ngf, nc, k).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)), k)

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(gd2.weights_init)
    return netG


def creaD(ngpu, ndf, nc, k, device):
     # Create the Discriminator
    netD = gd2.Discriminator(ngpu, ndf, nc, k).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netD.apply(gd2.weights_init)   
    return netD

#---------------------------------------------------------
def creaDeG(ngpu, nz, ngf, ndf, nc, k, device):
    print("Nuovo file")
    netG = creaG(ngpu, nz, ngf, nc, k, device)
    netD = creaD(ngpu,     ndf, nc, k, device)
    return netD, netG





def trainingStep(i,  data, 
                 real_label, fake_label, 
                 netD, netG, 
                 device, nz, optimizerD, optimizerG, criterion):
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
    D_x = output.mean().item()      # valore in output per la statistica sull'apprendimento 

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()   # valore in output per la statistica sull'apprendimento 
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake     # valore in output per la statistica sull'apprendimento 
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
    errG = criterion(output, label)  # valore in output per la statistica sull'apprendimento 
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()    # valore in output per la statistica sull'apprendimento 
    # Update G
    optimizerG.step()

    return errD, errG, D_x, D_G_z1, D_G_z2



def createDataloader(image_size, dataroot, n_samples, batch_size, workers):

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    trasf = transforms.Compose([   transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

    full_dataset = dset.ImageFolder(root=dataroot, transform=trasf)
    
    ## il numero di campioni puo' essere inferiore al totale
    # se n_samples > 0
    if n_samples > 0 :
        sottoinsieme = list(range(0, n_samples))
        dataset = torch.utils.data.Subset(full_dataset, sottoinsieme)
    else:
        dataset = full_dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  
                 shuffle=True, num_workers=workers)
  
    return dataloader


##======================================================
##======================================================
##======================================================
##======================================================
def main(pl, paramFile):

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    #print("Random Seed: ", manualSeed)
    #random.seed(manualSeed)
    #torch.manual_seed(manualSeed)

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

    dataloader = createDataloader(image_size, pl["dataroot"], pl["n_samples"], pl["batch_size"], pl["workers"])
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and pl["ngpu"] > 0) else "cpu")

    # crea le reti D e G e gli ottimizzatori================================================
    netD = creaD(pl["ngpu"], ndf, pl["nc"], k, device)
    optimizerD = optim.Adam(netD.parameters(), lr=pl["lrd"], betas=(pl["beta1"], pl["beta2"]))
    if pl["netD_checkpoint"] is not None:
        checkpoint = torch.load(pl["netD_checkpoint"])
        netD.load_state_dict(checkpoint['model_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])

    netG = creaG(pl["ngpu"], pl["nz"], ngf, pl["nc"], k, device)
    optimizerG = optim.Adam(netG.parameters(), lr=pl["lrd"], betas=(pl["beta1"], pl["beta2"]))
    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(pl["batch_size"], pl["nz"], 1, 1, device=device)

    if pl["netG_checkpoint"] is not None:
        checkpoint = torch.load(pl["netG_checkpoint"])
        netG.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
        fixed_noise = checkpoint['fixed_noise']
    
    # Print the model ==================================
    nomeFile = os.path.join(nomeDir, pl["nomeModello"]+"_architettura.txt")
    stringa = str(netD) +"\n\n"+ str(netG) 
    with open(nomeFile, "w") as text_file:
        text_file.write(stringa)
    print(stringa)
    #===================================================

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    print("Inizio apprendimento, tutti i dati saranno salvati in "+ nomeDir)
    # For each epoch
    for epoch in range(pl["num_epochs"]):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ## ADDESTRAMENTO DELLE RETI
            errD, errG, D_x, D_G_z1, D_G_z2 = trainingStep(i, data, 
                 real_label, fake_label, 
                 netD, netG, 
                 device, pl["nz"], 
                 optimizerD, optimizerG, 
                 criterion)
            ## Output training stats
            if i % 50 == 0:
                ss = stringaStato(epoch, pl["num_epochs"], i, 
                                  dataloader, errD, errG, D_x, D_G_z1, D_G_z2 )
                print(ss)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Fine dell'epoca --------------------------------------
        # salva un provino delle immagini generate ed i modelli relativi
        nomeFile = pl["nomeModello"]+ "_" +str(epoch)
        salvaCheckpoint(nomeDir, nomeFile, netD, netG, optimizerD, optimizerG, fixed_noise)

    # Fine del training =======================================
    nomeFile = pl["nomeModello"]+ "_"+"FINALE"
    salvaCheckpoint(nomeDir, nomeFile, netD, netG, optimizerD, optimizerG, fixed_noise)


    nomeFile = os.path.join(nomeDir, pl["nomeModello"] + "_"+pl["nomeFileLosses"][0])
    salvaCSV(G_losses, nomeFile)
    nomeFile = os.path.join(nomeDir, pl["nomeModello"] + "_"+pl["nomeFileLosses"][1])
    salvaCSV(D_losses, nomeFile)
    print("salvati i dati dell'apprendimento in ", nomeDir)



##======================================================
##======================================================
##======================================================
##======================================================
if __name__ == "__main__":
    inputFile = sys.argv[1]

    # Lettura parametri dal file yaml
    with open(inputFile) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_list = yaml.load(file, Loader=yaml.FullLoader)

    main(param_list, inputFile)
