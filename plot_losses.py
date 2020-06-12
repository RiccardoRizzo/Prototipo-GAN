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


def plot(dirModello):
    nomeFileLosses = findLossFiles(dirModello)
    #print(nomeFileLosses)

    nomeFile = os.path.join(dirModello, nomeFileLosses[0])
    #print(nomeFile)
    with open(nomeFile, 'r') as f:
        reader = csv.reader(f)
        G_losses = list(reader)
        
    nomeFile = os.path.join(dirModello, nomeFileLosses[1])
    #print(nomeFile)
    with open(nomeFile, 'r') as f:
        reader = csv.reader(f)
        D_losses = list(reader)
        
    G_losses = np.array([float(x) for x in  G_losses[0] ])
    D_losses = np.array([float(x) for x in  D_losses[0] ])

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    dirModello = sys.argv[1]
    plot(dirModello)