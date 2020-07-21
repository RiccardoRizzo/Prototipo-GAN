import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import sys
import os

import glob


def findLossFiles(dirModello):
    os.chdir(dirModello)
    suffix = "*losses.csv"
    listaFile = []
    for file in glob.glob(suffix):
        listaFile.append(file)
    os.chdir("..")
    print("trovati i file: " + " ".join(listaFile))
    return listaFile

#----------------------------------------------------
def loadLoss(nomefile):
    with open(nomefile, 'r') as f:
        buf = f.readlines()
    buf = [float(x.strip()) for x in buf]
    loss = np.array(buf)

    return loss

#---------------------------------------------------
def plotLosses(dirModello):

    nomeFileLosses = findLossFiles(dirModello)
    #print(nomeFileLosses)

    nomeFile = os.path.join(dirModello, nomeFileLosses[0])
    print(nomeFile)
    G_losses = loadLoss(nomeFile)
            
    nomeFile = os.path.join(dirModello, nomeFileLosses[1])
    print(nomeFile)
    D_losses = loadLoss(nomeFile)

    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":

    try:
        dirModello = sys.argv[1]
        plotLosses(dirModello)
    except:
        print('manca la directory che contiene i file di nome *losses.csv')
        sys.exit(1)  # abort
