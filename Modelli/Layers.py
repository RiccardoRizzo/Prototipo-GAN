import torch.nn as nn

# # Parameters to define the model.
# params = {
#     'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
#     'nz' : 100,# Size of the Z latent vector (the input to the generator).
#     'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
#     'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
#       }

#===============================================================================
kernel_size = 4
stride = 2
padding = 1

    # RIGUARDO LE DIMENSIONI ----------------------------------------------------
    # In nn.Conv2d(ndf * d_in, ndf * d_out, 4, 2, 1, bias=False)
    # kernel_size = 4, stride = 2, padding = 1
    # se kernel size = stride + 2* padding (come e') allora la dimensione di uscita della immagine e'
    # H_out = H_in / stride
    # W_out = W_out / stride
    #
    # Anche in nn.ConvTranspose2d(ngf * d_in, ngf * d_out, 4, 2, 1, bias=False))
    # H_out = H_in * stride
    # W_out = W_out * stride

#------------------------
def DisLayerSN_d(ndf, k):
    """
    Layer che usa la spectral norm
    """
    d_in = 2**k 
    d_out = 2**(k+1)

    out = nn.Sequential(nn.utils.spectral_norm(
                        nn.Conv2d(ndf*d_in, ndf*d_out, kernel_size, stride=stride, padding=padding, bias=False)),                        
                        nn.Dropout2d(),
                        nn.BatchNorm2d(ndf * d_out), 
                        nn.LeakyReLU(0.2, inplace=True) )
    return out

#------------------------
def DisLayerSN(ndf, k):
    """
    Layer che usa la spectral norm
    """
    d_in = 2**k 
    d_out = 2**(k+1)

    out = nn.Sequential(nn.utils.spectral_norm(
                        nn.Conv2d(ndf*d_in, ndf*d_out, kernel_size, stride=stride, padding=padding, bias=False)),                        
                        nn.BatchNorm2d(ndf * d_out), 
                        nn.LeakyReLU(0.2, inplace=True) )
    return out
    
#------------------------
def DisLayer(ndf, k):
    d_in = 2**k 
    d_out = 2**(k+1)

    out = nn.Sequential(nn.Conv2d(ndf*d_in, ndf*d_out, kernel_size, stride=stride, padding=padding, bias=False), 
                        nn.BatchNorm2d(ndf * d_out), 
                        nn.LeakyReLU(0.2, inplace=True) )
    return out

#===============================================================================

def GenLayerSN(ngf, k):
    """
    Layer che usa la spectral norm
    """
    d_in = 2**k 
    d_out = 2**(k-1)
    out = nn.Sequential( nn.utils.spectral_norm(
                         nn.ConvTranspose2d(ngf * d_in, ngf * d_out, kernel_size, stride, padding, bias=False)),
                         nn.BatchNorm2d(ngf * d_out),
                         nn.ReLU(True) )
    return out

#------------------------
def GenLayer(ngf, k):
    d_in = 2**k 
    d_out = 2**(k-1)
    out = nn.Sequential( nn.ConvTranspose2d(ngf * d_in, ngf * d_out, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm2d(ngf * d_out),
                         nn.ReLU(True) )
    return out