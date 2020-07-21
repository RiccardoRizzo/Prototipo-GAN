
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import sys
import os

import glob



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
    D_real = netD(real_cpu).view(-1)

    ## Train with all-fake batch
    # Generate batch of latent vectors
    z = torch.randn(b_size, nz, 1, 1, device=device)

   # Generate fake image batch with G
    G_sample = netG(z)
    # Classify all fake batch with D
    D_fake = netD(G_sample.detach()).view(-1)

    label.fill_(fake_label)

    D_loss = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2) )

    # Calculate the gradients for this batch
    D_loss.backward()
    optimizerD.step()


    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    D_fake = netD(G_sample).view(-1)

    G_loss = 0.5 * torch.mean((D_fake -1)**2 )
    
    G_loss.backward()
    optimizerG.step()
    
    out = [D_loss, G_loss]
    return out


#---------------------------------------------------------
def stringaStato(outTR):
    out = ('Loss_D: %.4f\tLoss_G: %.4f'
                    % (
                        # Loss_D    Loss_G
                        outTR[0].item(), outTR[1].item()
                        )
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

