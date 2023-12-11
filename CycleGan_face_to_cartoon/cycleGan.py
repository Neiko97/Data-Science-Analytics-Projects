import torch
from torch import nn


#######################
###### Generator ######
#######################
class FeatureMapBlock(nn.Module):
    '''
    The first and final layer of the Generator - 
    maps each the output to the desired number of output channels
    Values:
        in_channels: the number of input channels
        out_channels: the number of output channels
    '''
    def __init__(self, in_channels: int, out_channels: int, first: bool,  **kwargs):
        super(FeatureMapBlock, self).__init__()
        if first:
            self.conv = nn.Sequential(
                            nn.Conv2d(
                                in_channels, 
                                out_channels, 
                                padding_mode='reflect',
                                **kwargs
                            ),
                            nn.InstanceNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                        )
        else: 
            self.conv = nn.Conv2d(
                            in_channels, 
                            out_channels, 
                            padding_mode='reflect',
                            **kwargs
                        )
                           
                       


    def forward(self, x):
        return self.conv(x)

class ResidBlock(nn.Module):
    """
    The residual block plays a crucial role in the CycleGAN (Generative Adversarial Network) 
    architecture, serving as the transformative element for images. In the context of CycleGAN, 
    the residual block can be likened to the transformer, responsible for altering and enhancing 
    the characteristics of input images. This block is instrumental in facilitating the 
    cycle-consistency process.
    Args:
        in_channels: the number of input channels
        out_channels: the number of output channels
    """

    def __init__(self, in_channels: int, **kwargs):
        super(ResidBlock, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        self.conv2= nn.Sequential(
                nn.Conv2d(in_channels, in_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(in_channels),
                nn.Identity()
            )

    def forward(self, x):
        temp = x.clone()
        x0 = self.conv1(temp)
        return temp +  self.conv2(x0)
    

class ContractingBlockGen(nn.Module):
    """ 
    The contracting block within the CycleGAN architecture serves the essential purpose of 
    contracting the image tensor, leading to the extraction of a latent space representation
    Args:
        in_channels: the number of input channels
        out_channels: the number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ContractingBlockGen, self).__init__()
        self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        return self.down(x)


class ExpandingBlock(nn.Module):
    """ 
    The expanding block within the CycleGAN architecture serves the essential purpose of 
    expanding the latent space back to the actual image tensor size.
    Args:
        in_channels: the number of input channels
        out_channels: the number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ExpandingBlock, self).__init__()
        self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, padding_mode="zeros", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        return self.up(x)


class Generator(nn.Module):
    """
    The Generator puts all the components together in order to have a single streamline. 
    This class creates the fake images. 
    """

    def __init__(self, input_channels: int, output_channels: int, hidden_channels: int = 64, num_residuals: int = 6):
        super(Generator, self).__init__()
        self.init = FeatureMapBlock(
                        input_channels,
                        hidden_channels,
                        first = True,
                        kernel_size=7,
                        stride=1,
                        padding=3
                    )

        self.downsampling = nn.ModuleList([
                                ContractingBlockGen(
                                    hidden_channels, 
                                    hidden_channels * 2, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1
                                ),
                                ContractingBlockGen(
                                    hidden_channels * 2, 
                                    hidden_channels * 4, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1
                                )
                            ])
        self.residual = nn.Sequential(
            *[ResidBlock(hidden_channels * 4, kernel_size = 3, padding=1) for _ in range(num_residuals)]
        )

        self.upsampling = nn.ModuleList(
            [
                ExpandingBlock(
                    hidden_channels * 4,
                    hidden_channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ExpandingBlock(
                    hidden_channels * 2,
                    hidden_channels * 1,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            ]
        )

        self.last = FeatureMapBlock(
                        hidden_channels * 1,
                        output_channels,
                        first = False,
                        kernel_size=7,
                        stride=1,
                        padding=3
                    )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.init(x)
        
        for layer in self.downsampling:
            x = layer(x)

        x = self.residual(x)

        for layer in self.upsampling:
            x = layer(x)

        x = self.last(x)

        return self.tanh(x)


###########################
###### Discriminator ######
###########################

class ContractingBlockDisc(nn.Module):
    """ 
    The contracting block within the CycleGAN architecture serves the essential purpose of 
    contracting the image tensor, leading to the extraction of a latent space representation
    Args:
        in_channels: the number of input channels
        out_channels: the number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ContractingBlockDisc, self).__init__()
        self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
    def forward(self, x):
        return self.down(x)

class Discriminator(nn.Module):
    '''
    The discriminator class is used to distinguish between the real images and fake images. It is used 
    to train the generator as well as itself. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.init = self.initial_layer =nn.Sequential(
                                            nn.Conv2d(
                                                input_channels,
                                                hidden_channels,
                                                kernel_size=4,
                                                stride=2,
                                                padding=1,
                                                padding_mode="reflect",
                                            ),
                                            nn.LeakyReLU(0.2, inplace=True),
                                        )
        self.contract1 = ContractingBlockDisc(hidden_channels,hidden_channels * 2, kernel_size=4)
        self.contract2 = ContractingBlockDisc(hidden_channels * 2, hidden_channels * 4,kernel_size=4)
        self.contract3 = ContractingBlockDisc(hidden_channels *4, hidden_channels * 8, kernel_size=4)
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        x0 = self.init(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        return self.final(x3)
      