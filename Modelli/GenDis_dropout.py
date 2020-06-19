import torch.nn as nn
import Layers as ll

# # Parameters to define the model.
# params = {
#     'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
#     'nz' : 100,# Size of the Z latent vector (the input to the generator).
#     'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
#     'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
#       }

#===============================================================================



class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, k):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        layers = []

        layers.append(nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False) )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 64 x 64


        for i in range(k):
            layers.append(ll.DisLayer(ndf, i))

        d_out = 2**k
        layers.append(nn.Conv2d(ndf * d_out, 1, 4, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        # state size. 1
        
        self.main = nn.ModuleList(layers)



    def forward(self, x):
        y = x
        for i in range(len(self.main)):
            y = self.main[i](y)
        return y


#===============================================================================

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, k):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        layers = []

 
        d_in = 2**k
        layers.append( nn.ConvTranspose2d( nz, ngf * d_in, 4, 1, 0, bias=False) )
        layers.append( nn.BatchNorm2d(ngf * d_in) )
        layers.append( nn.ReLU(True) )
        # state size. (ngf*16) x 4 x 4
            
        
        for i in range(k):
            n = k-i 
            layers.append( ll.GenLayerDropout(ngf, n) )

            
        layers.append(nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False) )
        layers.append(nn.Tanh() )
            # state size. (nc) x 128 x 128
 

        self.main = nn.ModuleList(layers)
        

    def forward(self, x):
        y = x
        for i in range(len(self.main)):
            y = self.main[i](y)
        return y



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
