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

###=============================================
###  DEFINIZIONE MODELLO GAN E ALGORITMO 
###  DI TRAINING ===============================
sys.path.append("./Modelli")

import GenDis_b4_SA as gd2
import ltr_DCGAN as tr
# import ltr_LSGAN as tr # RICORDARSI CHE DISC. DEVE ESSERE SENZA SIGMOIDE IN OUT
###=============================================

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

def salvaImmagini(nomeDir, nomeFile, netG, fixed_noise):
    with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            nomeFileImage = os.path.join(nomeDir, nomeFile +".jpg")
            saveFakeImages(fake, nomeFileImage)


#---------------------------------------------------
def salvaCheckpoint(nomeDir, nomeFile, netD, netG, optimizerD, optimizerG, fixed_noise):
    
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

#---------------------------------------------------
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

#---------------------------------------------------
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


def creaDocu(pl, paramFile):
    # e copia i file usati (questo compreso)  nella directory 
    # dell'esperimento per riferimenti futuri
    today = datetime.now()
    nomeDir = "./" + pl["nomeModello"] + "_" +today.strftime('%Y_%m_%d_%H_%M')
    os.mkdir(nomeDir)

    nomeDirPy = nomeDir + "/sources"
    os.mkdir(nomeDirPy)
    # copio il file di parametri nella dir dell'esperimento
    newPath = shutil.copy(paramFile, nomeDir)
    # copio il file dei modelli
    newPath = shutil.copy(gd2.__file__, nomeDirPy)
    # copio il file del training
    newPath = shutil.copy(tr.__file__, nomeDirPy)
    # copio questo file nella directory
    newPath = shutil.copy(__file__, nomeDirPy)

    # zippa i sorgenti
    shutil.make_archive(nomeDir+"/sources", 'zip', nomeDirPy)
    # cancella la dir
    shutil.rmtree(nomeDirPy)

    return nomeDir


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

    # number of layers
    k = int(math.log(pl["image_size"], 2)) - 3
    # k = pl["k"]


    # creazione del dataloader =============================================================
    dataloader = createDataloader(pl["image_size"], pl["dataroot"], pl["n_samples"], pl["batch_size"], pl["workers"])
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and pl["ngpu"] > 0) else "cpu")

    # crea le reti D e G e gli ottimizzatori================================================
    netD = creaD(pl["ngpu"], pl["ndf"], pl["nc"], k, device)
    optimizerD = optim.Adam(netD.parameters(), lr=pl["lrd"], betas=(pl["beta1"], pl["beta2"]))
    if pl["netD_checkpoint"] is not None:
        checkpoint = torch.load(pl["netD_checkpoint"])
        netD.load_state_dict(checkpoint['model_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])

    netG = creaG(pl["ngpu"], pl["nz"], pl["ngf"], pl["nc"], k, device)
    optimizerG = optim.Adam(netG.parameters(), lr=pl["lrg"], betas=(pl["beta1"], pl["beta2"]))
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(pl["batch_size"], pl["nz"], 1, 1, device=device)

    if pl["netG_checkpoint"] is not None:
        checkpoint = torch.load(pl["netG_checkpoint"])
        netG.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
        fixed_noise = checkpoint['fixed_noise']
    
    # creazione della directory dell'esperimento ==================================
    nomeDir = creaDocu(pl, paramFile)
    # Print the model ==================================
    nomeFile = os.path.join(nomeDir, pl["nomeModello"]+"_architettura.txt")
    stringa = str(netD) +"\n\n"+ str(netG) 
    with open(nomeFile, "w") as text_file:
        text_file.write("############################\n")
        text_file.write("File rete in " + gd2.__file__ + "\n\n\n")
        text_file.write(stringa)
        text_file.write("\nAlgoritmo apprendimento in " + tr.__file__)
    print(stringa)
    #===================================================

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0


    print("Inizio apprendimento, tutti i dati saranno salvati in "+ nomeDir)
    
    # crea i file per le perdite e scrive il riferimento per i minibatch =====================
    nomeFile_G_losses = os.path.join(nomeDir, pl["nomeModello"] + "_"+pl["nomeFileLosses"][0])
    nomeFile_D_losses = os.path.join(nomeDir, pl["nomeModello"] + "_"+pl["nomeFileLosses"][1])
    # salva il numero di minibatch per epoca per creare dei riferimenti
    with open(nomeFile_G_losses, 'a+', newline='') as myfile:
    #with open(nomefile, 'w', newline='') as myfile:
        stringa = "# " + str( len(dataloader) ) + "\n"
        myfile.write(stringa)
        
    with open(nomeFile_D_losses, 'a+', newline='') as myfile:
        stringa = "# " + str( len(dataloader) ) + "\n"
        myfile.write(stringa)

    # INIZIO APPRENDIMENTO ===================================================================
    # For each epoch
    for epoch in range(pl["num_epochs"]):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            #------------------------------------
            # print("ADDESTRAMENTO DELLE RETI")
            datiTR = tr.trainingStep(i, data, 
                 real_label, fake_label, 
                 netD, netG, 
                 device, pl["nz"], 
                 optimizerD, optimizerG, 
                 criterion)
            ## Output training stats
            if i % 50 == 0:
                # stringa relativa ad epoche e dati
                oo = '[%d/%d][%d/%d]\t' % (epoch, pl["num_epochs"],  i, len(dataloader))
                # stringa relativa allo stato dell'apprendimento
                ss = tr.stringaStato( datiTR )
                print(oo + ss)

            # Save Losses for plotting later
            # l'output dal training step [errD, errG, D_x, D_G_z1, D_G_z2]
            tr.D_losses.append(str(datiTR[0].item()))
            tr.G_losses.append(str(datiTR[1].item()))

            torch.cuda.empty_cache()

        # Fine dell'epoca --------------------------------------
        # salva un provino delle immagini generate ed i modelli relativi
        nomeFile = pl["nomeModello"]+ "_" +str(epoch)

        salvaImmagini(nomeDir, nomeFile, netG, fixed_noise)

        #tr.salvaCSV(tr.G_losses, nomeFile_G_losses)
        tr.salvaLoss(tr.G_losses, nomeFile_G_losses)
        tr.G_losses = []
        #tr.salvaCSV(tr.D_losses, nomeFile_D_losses)
        tr.salvaLoss(tr.D_losses, nomeFile_D_losses)
        tr.D_losses = []
        
        # salva i modelli ogni cadenza_epoche   
        if epoch % pl["cadenza_epoche"] == 0:
            salvaCheckpoint(nomeDir, nomeFile, netD, netG, optimizerD, optimizerG, fixed_noise)

    # FINE APPRENDIMENTO =====================================================================
    nomeFile = pl["nomeModello"]+ "_"+"FINALE"
    salvaCheckpoint(nomeDir, nomeFile, netD, netG, optimizerD, optimizerG, fixed_noise)
    salvaImmagini(nomeDir, nomeFile, netG, fixed_noise)

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
