################################################
# CALCOLO DELLE DISTANZE FRECHET-INCEPTION
# FRA DUE LISTE DI DIRECTORY
################################################


import yaml
import sys
import torch
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os


sys.path.append("./pytorch_fid")
import fid_score as fid

sys.path.append("./confusion_matrix-master")
import cf_matrix as cfm


#==================================================================================
def visualizza (mat, pl):
    """
    ARGOMENTI DELLA FUNZIONE make_confusion_matrix

    cf:           confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix. 
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'

    See http://matplotlib.org/examples/color/colormaps_reference.html
    
    title:         Title for the heatmap. Default is None.
    """

    cfm.make_confusion_matrix(  cf = mat, 
                                group_names = pl["dir1"], 
                                categories = pl["dir2"], 
                                normalize = False, 
                                cbar = True,
                                xyticks = True,
                                xyplotlabels = True, 
                                sum_stats = False, 
                                title = "Dissimilarita'"
                                 )





#==================================================================================
def main(pl, nome_file_params):
    # calcola la matrice di confusione
    mat = calcola_confusion_matrix(pl)

    # leva il path dal nome file di input
    base=os.path.basename(nome_file_params)
    # elimina dal nomefile_params la estensione
    nome_file, _ = os.path.splitext(base)

    # salva la matrice
    nome_file_mat = nome_file + ".csv"
    # salva la matrice 
    np.savetxt(nome_file_mat, mat, delimiter=",")

    # trasforma in un array numpy
    #mat = genfromtxt(nome_file, delimiter=',')
    mat = np.array(mat)
    # esegue la visulazzazione
    visualizza(mat, pl)
    # salva l'immagine
    nome_immagine = nome_file + ".png"
    plt.savefig(nome_immagine, bbox_inches='tight')
    # opzionalmnte mostra la immagine
    plt.show()
    
#==================================================================================
def calcola_statistiche_del_path(path, batch_size, dims, device):
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)

    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    m, s = fid._compute_statistics_of_path(path, model, batch_size, dims, device)

    return m, s


#==================================================================================
def calcola_confusion_matrix(pl):

    batch_size = pl["batch_size"]
    device = pl["device"]
    dims = pl["dims"]
    
    
    stats1 = {}
    # calcolo le statistiche della directory
    for ld1 in pl["dir1"]:
        path_ld1 = pl["prefix1"] + ld1
        print("Calcolo statistica directory: " + ld1)
        m, s = calcola_statistiche_del_path(path_ld1,  batch_size, dims, device)
        stats1[ld1] = (m, s)

    stats2 = {}
    for ld2 in pl["dir2"]:
        path_ld2 = pl["prefix2"] + ld2
        print("Calcolo statistica directory: " + ld2)
        m, s = calcola_statistiche_del_path(path_ld2, batch_size, dims, device)
        stats2[ld2] = (m, s)


    matrice = np.zeros((len(pl["dir1"]), len(pl["dir2"])))
    for i in range(len(pl["dir1"])):
        ld1 = pl["dir1"][i]
        for j in range(i, len(pl["dir2"])):
            ld2 = pl["dir2"][j]
            dd = fid.calculate_frechet_distance(stats1[ld1][0], stats1[ld1][1], 
                                                stats2[ld2][0], stats2[ld2][1])
            print(" distanza fra " + ld1 + " e " + ld2 + " = " +  str(dd) )
            matrice[i][j] = dd
            matrice[j][i] = dd
    
    return matrice


#==================================================================================
# esempio di riga di comandoi
# in questo modo esegue il calcolo e la visulazzazione 
#
# fid_dist_2_lists.py parametri.yaml -c
#
# in questo modo esegeu solo la visualizzazione 
# della matrice parametri.csv salvata in un file csv
# i nomi delle categorie sono nel file parametri.yaml
#
# fid_dist_2_lists.py parametri.yaml -v


if __name__ == "__main__":
    inputFile = sys.argv[1]
    
    mode = sys.argv[2]
    
    # esegue il calcolo e la visulaizzazione
    if mode == "-c":
        # Lettura parametri dal file yaml
        with open(inputFile) as file:
            param_list = yaml.load(file, Loader=yaml.FullLoader)

        main(param_list, inputFile)

    # esegue solo la visulaizzazione
    if mode == "-v":
        # Lettura parametri dal file yaml
        with open(inputFile) as file:
            param_list = yaml.load(file, Loader=yaml.FullLoader)

        # carica la matrice dal file csv
        # 
        nome_file = os.path.splitext(inputFile)[0] + ".csv"
        mat = np.loadtxt(open(nome_file, "rb"), delimiter=",")
        print(mat.size)
        visualizza(mat, param_list)
        # opzionalmnte mostra la immagine
        plt.show()
    

