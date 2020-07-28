import torch.nn as nn
#import Layers as ll
import self_attention as sa
import torch


# # Parameters to define the model.
# params = {
#     'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
#     'nz' : 100,# Size of the Z latent vector (the input to the generator).
#     'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
#     'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
#       }

#===============================================================================

###############################################
## CAMBIARE PER OTTENERE LA MOLTIPLICAZIONE 
## PER 4 NELLA CREAZIONE DELLE IMMAGINI
# -------------------------------------------- 
kernel_size = 6 
stride = 4
padding =1
# Nella costruzione della rete occorre ricordare che
# ------- Per il Generatore -------------------
# se risulta Kernel_size = stride + 2 * padding
# allora    H_out = H_in * stride
#           W_out = W_in * stride
# ------- Per il discriminatore ---------------
# se risulta Kernel_size = stride + 2 * padding
# allora    H_out = H_in / stride
#           W_out = W_in / stride
###############################################

#################################################################################################
def DisLayerSN_d(ndf, k):
    """
    Layer che usa la spectral norm
    """
    d_in = stride**k 
    d_out = stride**(k+1)

    out = nn.Sequential(nn.utils.spectral_norm(
                        nn.Conv2d(ndf*d_in, ndf*d_out, kernel_size, stride=stride, padding=padding, bias=False)),                        
                        nn.Dropout2d(),
                        nn.BatchNorm2d(ndf * d_out), 
                        nn.LeakyReLU(0.2, inplace=True) )
    return out

#---------------------------------------

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, k):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        layers = []

        layers.append(nn.Conv2d(nc, ndf, kernel_size, stride, padding=padding, bias=False) )
        layers.append(nn.LeakyReLU(0.2, inplace=True))


        #--------------------------------------------
        layers.append(DisLayerSN_d(ndf, 0))
        layers.append(DisLayerSN_d(ndf, 1))
        #--------------------------------------------

        d_out = stride**2

        #layers.append(nn.Dropout2d())
        layers.append(nn.Conv2d(ndf * d_out, 1, 2, stride, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        # state size. 1
        
        self.main = nn.ModuleList(layers)



    def forward(self, x):
        y = x
        #print("D_rete ingresso ", str(y.size()))
        for i in range(len(self.main)):
            #print("D:in ", str(y.size()))
            y = self.main[i](y)
            #print("D:out ", str(y.size()))
        return y




#################################################################################################
def GenLayerSN(ngf, k):
    """
    Layer che usa la spectral norm
    """
    d_in = stride**k 
    d_out = stride**(k-1)
    out = nn.Sequential( nn.utils.spectral_norm(
                         nn.ConvTranspose2d(ngf * d_in, ngf * d_out, kernel_size, stride, padding, bias=False)),
                         nn.BatchNorm2d(ngf * d_out),
                         nn.ReLU(True) )
    return out

#-----------------------------------------------

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, k):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        layers = []

        d_in = stride**2

        layers.append( nn.ConvTranspose2d( nz, ngf * d_in, kernel_size, stride, padding, bias=False) )
        layers.append( nn.BatchNorm2d(ngf * d_in) )
        layers.append( nn.ReLU(True) )

        #------------------------------------------      
        layers.append( GenLayerSN(ngf, 2) )
        layers.append( GenLayerSN(ngf, 1) ) 
        #------------------------------------------

        layers.append(sa.Self_Attn(ngf,"relu"))    
        
        layers.append(nn.ConvTranspose2d( ngf, nc, kernel_size, stride, padding, bias=False) )
        layers.append(nn.Tanh() )

        self.main = nn.ModuleList(layers)
        

    def forward(self, x):

        y = x
        #print("G_rete ingresso ", str(y.size()))
        for i in range(len(self.main)):
            #print("G:in ", str(y.size()))
            y = self.main[i](y)
            #print("G:out ", str(y.size()))
        return y



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
