import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, latent_size = 512):
        super(EncoderDecoder, self).__init__()
        self.latent_size = latent_size
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),   # 16x35x35 [(W−K+2P)/S]+1
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # 32x17x17
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x5x5    
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=0), # 256x3x3    
            nn.ReLU(),
            nn.Conv2d(256, 512, 2, stride=1, padding=0), # 512x2x2    
            nn.ReLU(),
            nn.Conv2d(512, 512, 2, stride=1, padding=0), # 512x1x1
            nn.ReLU(),
            nn.Flatten()  # Rappresentazione latente 512
        )
        # Decoder: parte da 512x1x1 e ricostruisce fino a 3x70x70
        self.Decoder = nn.Sequential( #  (W - 1)S -2P + (K - 1) + 1
            nn.ConvTranspose2d(512, 512, 2, stride=1, padding=0), # 256x2x2    
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=1, padding=0), # 256x3x3  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0), # 128x5x5  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),  # 64x9x9  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),   # 32x17x17
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=1),   # 16x35x35
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),    # 3x70x70
            #nn.Tanh()  # Per normalizzare l'output tra -1 e 1
        )

    def forward(self, x):
        ## encode ##
        latent = self.Encoder(x)  # compressed representation
        ## decode ##
        x = self.Decoder(latent.view(x.size(0),self.latent_size,1,1))
        return x