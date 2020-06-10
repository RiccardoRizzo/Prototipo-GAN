import torch
import torch.nn as nn
import torch.nn.functional as F


# # Parameters to define the model.
# params = {
#     "bsize" : 128,# Batch size during training.
#     'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
#     'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
#     'nz' : 100,# Size of the Z latent vector (the input to the generator).
#     'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
#     'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
#     'nepochs' : 10,# Number of training epochs.
#     'lr' : 0.0002,# Learning rate for optimizers
#     'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
#     'save_epoch' : 2}# Save step.

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        layers = []
        num_stages = params["num_stages"]
        dim = 2**num_stages
        # Input is the latent vector Z.
        ll= nn.Sequential(
            nn.ConvTranspose2d(params['nz'], params['ngf']*dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(params['ngf']*dim)
        )

        layers.append(ll)

        for i in range(num_stages, 0, -1):
            d_in = 2**i
            d_out = 2**(i-1)
            layer = nn.Sequential( 
                nn.ConvTranspose2d(params['ngf']*d_in, params['ngf']*d_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(params['ngf']*d_out)
            )

            layers.append(layer)

        # Input Dimension: (ngf) * 32 * 32
        layers.append( nn.ConvTranspose2d(params['ngf'], params['nc'], 4, 2, 1, bias=False) )
        #Output Dimension: (nc) x 64 x 64
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.net[:-1]:
            x = F.relu(l(x))
        x = F.tanh(self.net[-1](x))

        return x



# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        layers = []
        num_stages = params["num_stages"]
        # Input Dimension: (nc) x 64 x 64
        layers.append( nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False) )

        # Input Dimension: (ndf) x input_dim (=64 la prima volta)/d_in x input_dim (=64 la prima volta)/d_in
        for i in range(0, num_stages):
            d_in = 2**i
            d_out = 2**(i+1)
            layer = nn.Sequential( 
                    nn.Conv2d(params['ndf']*d_in, params['ndf']*d_out, 4, 2, 1, bias=False), 
                    nn.BatchNorm2d(params['ndf']*d_out)
                )
            layers.append(layer)

        # Input Dimension: (ndf*8) x 4 x 4
        dim = 2**num_stages
        layers.append(
            nn.Conv2d(params['ndf']*dim, 1, 4, 1, 0, bias=False)
        )

        self.net = nn.ModuleList(layers)



    def forward(self, x):
        for l in self.net[:-1]:
            x = F.leaky_relu(l(x), 0.2, True)
        x = F.sigmoid(self.net[-1](x))

        return x
