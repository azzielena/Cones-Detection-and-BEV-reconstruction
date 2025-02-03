import torch
import torch.nn as nn

# UNet architecture
def conv_layer(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    )

class UNet1024(nn.Module):
    def __init__(self):
        super(UNet1024, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input dim 256x256
        self.down_1 = conv_layer(3, 64) #128x128
        self.down_2 = conv_layer(64, 128) #64x64
        self.down_3 = conv_layer(128, 256) #32x32
        self.down_4 = conv_layer(256, 512) #16x16
        self.down_5 = conv_layer(512, 1024) #8x8
        
        #  (W - 1)S -2P + (K - 1) + 1
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024, 512)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.up_conv_2 = conv_layer(512, 256)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.up_conv_3 = conv_layer(256, 128)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128, 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1) # Changed output channels to 3
        self.output_activation = nn.Sigmoid()
                
    def forward(self, img):     #The print statements can be used to visualize the input and output sizes for debugging
        x1 = self.down_1(img) #256x256
        x2 = self.max_pool(x1)
        x3 = self.down_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_4(x6)
        x8 = self.max_pool(x7) #8x8
        x9 = self.down_5(x8)
    
        x = self.up_1(x9) # x.size() = bs, 512, h, w x7.size() = bs, 512, h, w #16x16
        x = self.up_conv_1(torch.cat([x, x7], 1)) # torch.cat([x, x7], 1).size() = bs, 1024, h, w
        x = self.up_2(x)
        x = self.up_conv_2(torch.cat([x, x5], 1)) 
        x = self.up_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x = self.up_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1)) #256x256
        
        x = self.output(x)
        return x