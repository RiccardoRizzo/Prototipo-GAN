# DCGAN :: modello a risoluzione maggiore

## 11 giugno 2020


## 10 giugno 2020

* train_DCGAN.py
  e' il programma la lanciare con argomento un file di parametri ( per esempio train_DCGAN.py parametri_BASE.yaml)

* Modelli
Contiene le reti; i file sono:

  * Layers.py :  il codice dei vari strati di G e D
  
  * GenDis***.py : le reti Generatore e Discriminatore costruite usando gli strati e altri moduli
  
  * spectral.py : codice della norma spettrale


Il file dei parametri non necessita di settare il numero di strati k
Il file parametri_BASE.yaml e' quello da clonare e modificare per gli esperimenti.
Tutti i commenti e le modifiche sostanziali vanno fatte prima in questo file, in modcontiene o da renderle permanenti. 

Viene creata una directory con dettagli e risultati del run
I risultati sono variabili, e' necessario fare altri esperimenti.

## 17 febbraio 2020

Lanciato su macchina Deep un apprendimento su celeba-HQ 1024 x 1024 per generare immagini 128 x 128.

GenDis5 funziona per 63, 128 e forse 256. Relazione fra k e dimensione della immagine

- k=5 immagine 256 x 256
- k=4 immagine 128 x 128
- k=3 immagine 64 x 64

Il file GenDis4 a 128 funziona, quindi modifico per fare il ciclo nella creazione degli strati

## 10 febbraio 2020

![dcgan_generator](/home/riccardo/Desktop/Link to Mia ProGAN/Inizio con una DCGAN/2== Modello a risoluzione maggiore/figure/dcgan_generator.png)

```python
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```



```
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)

```



## 5 febbraio 2020

dentro la dir [daverificare](/home/riccardo/Desktop/Link to Mia ProGAN/Inizio con una DCGAN/daverificare) c'e' la [DCGAN modificata](/home/riccardo/Desktop/Link to Mia ProGAN/Inizio con una DCGAN/daverificare/dcgan2.py) che volevo fare

competizioni pee la classificazione di immagini istologiche:

- [kaggle1](https://www.kaggle.com/c/histopathologic-cancer-detection)
- [grand challenge](https://grand-challenge.org/)

##### [Articolo su medical imagining generation](https://paperswithcode.com/task/medical-image-generation)

Vediamo per quale motivo non riesco a visualizzare le immagini in output. 

Per la visualizzazione di una singola immagine generata dalla DCGAN nella [directory](/home/riccardo/Desktop/Link to Mia ProGAN/Inizio con una DCGAN) il codice sotto funziona

```python
from PIL import Image

# GENERAZIONE DELLA IMMAGINE
# generazione del rumore (seme)
fixed_noise = torch.randn(1, nz, 1, 1, device=device)
# generazione immagine
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
# prelievo della immagine (corrispondente a unsqueeze)    
img = fake[0]
# # trasformazione in array (H x W x C)
img2 = np.array( np.transpose(img,(1,2,0)) )

# normalizzazione
min_ = np.min(img2)
max_ = np.max(img2)

img2 = np.add(img2, -min_)
img2 = np.divide(img2, (max_-min_ +1e-5))
img2 = np.multiply(img2,255.0)

# trasformazione 
img2 = np.uint8(img2)

img2 = Image.fromarray(img2)
img2.show()
```







## 4 febbraio 2020

Seguo il [tutorial di pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) 

[Paper originale](https://arxiv.org/pdf/1511.06434.pdf)

al momento metto tutto in un notebook jupyter.



Tasformato il notebbok in un eseguibile train_sempliceDCGAN.py, sulla macchina  Deep da' questo errore:

```
/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Discriminator. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Sequential. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Conv2d. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type LeakyReLU. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type BatchNorm2d. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Sigmoid. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Generator. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type ConvTranspose2d. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type ReLU. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "

/home/ricrizzo/anaconda/envs/pt/lib/python3.7/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Tanh. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "
salvato il modello in  Prova1

```

Il salvataggio della rete   e' fatto con

```python
torch.save(netD, nomeModello +"__D.pth" )
torch.save(netG, nomeModello +"__G.pth" )
```

ed il ripristino dovrebbe essere con:

```python
location = "cuda" if torch.cuda.is_available() else "cpu"
model=torch.load(nomeModello, map_location = location)
```





------

Secondo [Stackoverflow](https://stackoverflow.com/questions/52277083/pytorch-saving-model-userwarning-couldnt-retrieve-source-code-for-container-of) la soluzione e'

Saving

```python
torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
```

Loading

```python
model = describe_model()
checkpoint = torch.load('checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
```



