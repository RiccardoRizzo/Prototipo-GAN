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

    # elimino il primo valore
    num_minibatch = int(buf[0]. strip("#"))
    buf = buf[1:]

    buf = [float(x.strip()) for x in buf]
    loss = np.array(buf)

    return loss, num_minibatch

#---------------------------------------------------
def plotLosses(dirModello):

    nomeFileLosses = findLossFiles(dirModello)
    #print(nomeFileLosses)

    nomeFile = os.path.join(dirModello, nomeFileLosses[0])
    print(nomeFile)
    G_losses, n_b = loadLoss(nomeFile)
  
            
    nomeFile = os.path.join(dirModello, nomeFileLosses[1])
    print(nomeFile)
    D_losses, n_b = loadLoss(nomeFile)

    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")

    xcoords = [x  for x in range(n_b, len(G_losses)+1, n_b) ]


    for xc in xcoords:
        plt.axvline(x=xc,  color='k', linestyle='--')


    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":

    try:
        dirModello = sys.argv[1]
        plotLosses(dirModello)

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        #print('manca la directory che contiene i file di nome *losses.csv')
        sys.exit(1)  # abort
