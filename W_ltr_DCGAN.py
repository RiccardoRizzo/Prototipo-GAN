import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import sys
import os

import glob

#########################################################################
#
#       WASSERSTEIN GAN
#   * No log in the loss. The output of D is no longer a probability, 
#       hence we do not apply sigmoid at the output of D
#   * Clip the weight of D          <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   * Train D more than G           <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
#   * Use RMSProp instead of ADAM
#   * Lower learning rate, the paper uses
#           \alpha = 0.00005
#
#########################################################################

def trainingStep(i,  data, 
                 real_label, fake_label, 
                 netD, netG, 
                 device, nz, optimizerD, optimizerG, criterion):

    #>>>> ADDESTRARE IL DISCRIMINATORE MOLTO DI PIU' DEL GENERATORE
    for _ in range(5):
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

        #>>> CLIPPING DEI PESI DEL DISRIMINATORE
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

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

    outTR = [errD, errG, D_x, D_G_z1, D_G_z2]

    return outTR





#---------------------------------------------------------
def stringaStato(outTR):
    """
    Serve a stampare una stringa con il report relativo al passo di apprendimento
    definito sopra
    """
    out = ('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                        # Loss_D        Loss_G
                        (outTR[0].item(), outTR[1].item(), 
                        #D(x)     D(G(z))
                        outTR[2], outTR[3], outTR[4])
            )
    return out



#####################################################
"""
Strutture per la memorizzazione dell'errore durante l'apprendimento
"""
G_losses = []
D_losses = []


#---------------------------------------------------
def salvaLoss(lista, nomefile):
    """
    Salva una delle strutture di sopra in un file csv
    """
    with open(nomefile, 'a+', newline='') as myfile:
    #with open(nomefile, 'w', newline='') as myfile:
        stringa = "\n".join(lista)
        myfile.write(stringa)
        myfile.write("\n")


