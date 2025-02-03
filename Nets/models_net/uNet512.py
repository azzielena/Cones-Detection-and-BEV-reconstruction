import torch
import torch.nn as nn

#Define the UNet architecture
def conv_layer(input_channels, output_channels):     #This is a helper function to create the convolutional blocks
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    )
    return conv
 
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
 
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input dim 70X70
        self.down_1 = conv_layer(3, 64) #35X35
        self.down_2 = conv_layer(64, 128) #17X17
        self.down_3 = conv_layer(128, 256) #8X8
        self.down_4 = conv_layer(256, 512) #4X4
 
 
        #  (W - 1)S -2P + (K - 1) + 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.up_conv_1 = conv_layer(512, 256)
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.up_conv_2 = conv_layer(256, 128)
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(128, 64)
 
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1) # Changed output channels to 3
        self.output_activation = nn.Sigmoid()
 
    def forward(self, img):     #The print statements can be used to visualize the input and output sizes for debugging
        x1 = self.down_1(img)
        #print(x1.size()) #70X70
        x2 = self.max_pool(x1)
        #print(x2.size()) #35X35
        x3 = self.down_2(x2)
        #print(x3.size())
        x4 = self.max_pool(x3)
        #print(x4.size()) #17X17
        x5 = self.down_3(x4)
        #print(x5.size())
        x6 = self.max_pool(x5)
       # print(x6.size()) #8X8
        x7 = self.down_4(x6)
        #print(x7.size())  #[512,8,8]
 
 
        x = self.up_1(x7)
        #print(x.size())
        x = self.up_conv_1(torch.cat([x, x5], 1))
        #print(x.size())
        x = self.up_2(x)
        #print(x.size())
        x = self.up_conv_2(torch.cat([x, x3], 1))
        #print(x.size())
        x = self.up_3(x)
        #print(x.size())
        x = self.up_conv_3(torch.cat([x, x1], 1))
        #print(x.size())
 
        x = self.output(x)
        #x = self.output_activation(x)
        #print(x.size())
        return x