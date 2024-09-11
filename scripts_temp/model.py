import torch
import torch.nn as nn

class Conv_Unit(nn.Module):                                                                     #condensing a convolutional unit

    def __init__(self, in_dim, out_dim, is_branch):

        super(Conv_Unit, self).__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size = (3, 3), padding = 1)               #convolution
        self.bn = nn.BatchNorm2d(out_dim)                                                       #batch normalization
        self.relu = nn.LeakyReLU()                                                              #leaky ReLU
        self.is_branch = is_branch

    def forward(self, flow):

        flow = self.conv(flow)

        if self.is_branch == 0:
            flow = self.bn(flow)
            flow = self.relu(flow)

        return flow
    
    
class Pooling_Unit(nn.Module):                                                                  #condensing a pooling unit

    def __init__(self, in_dim):

        super(Pooling_Unit, self).__init__()

        self.bn = nn.BatchNorm2d(in_dim)                                                        #batch normalization
        self.relu = nn.LeakyReLU()                                                              #leaky ReLU
        self.max_pool = nn.MaxPool2d(2, stride = (2, 2))                                        #max pooling

    def forward(self, flow):
        flow = self.bn(flow)
        flow = self.relu(flow)
        flow = self.max_pool(flow)


class Trans_Conv_Unit(nn.Module):                                                               #transposed convolutional unit

    def __init__(self, in_dim, out_dim):

        super(Trans_Conv_Unit, self).__init__()

        self.conv_t = nn.ConvTranspose2d(in_dim, out_dim, kernel_size = (3, 3), padding = 1)    #transposed convolution
        self.bn = nn.BatchNorm2d(out_dim)                                                       #batch normalization
        self.relu = nn.LeakyReLU()                                                              #leaky ReLU

    def forward(self, flow):
        
        flow = self.conv_t(flow)
        flow = self.bn(flow)
        flow = self.relu(flow)

        return flow
    
        
class U_Net(nn.Module):
    
    def __init__(self):

        super(U_Net, self).__init__()

        self.encoder_1 = nn.Sequential(
            
            Conv_Unit(1, 64, 0),                     #1 conv
            Conv_Unit(64, 64, 0),                    #2 conv
            Conv_Unit(64, 64, 1)                     #3 conv

        )

        self.encoder_2 = nn.Sequential(

            Pooling_Unit(64),                        #1 max pool
            Conv_Unit(64, 128, 0),                   #4 conv
            Conv_Unit(128, 128, 0),                  #5 conv
            Conv_Unit(128, 128, 1)                   #6 conv

        )

        self.encoder_3 = nn.Sequential(

            Pooling_Unit(128),                       #2 max pool 
            Conv_Unit(128, 256, 0),                  #7 conv
            Conv_Unit(256, 256, 0),                  #8 conv
            Conv_Unit(256, 256, 1)                   #9 conv
        
        )

        self.encoder_4 = nn.Sequential(

            Pooling_Unit(256),                       #3 max pool
            Conv_Unit(256, 512, 0),                 #10 conv
            Conv_Unit(512, 512, 0),                 #11 conv
            Conv_Unit(512, 512, 1),                 #12 conv
        
        )

        self.bottleneck_decoder_1 = nn.Sequential(

            Pooling_Unit(512),                       #4 max pool
            Conv_Unit(512, 1024, 0),                #13 conv
            Conv_Unit(1024, 1024, 0),               #14 conv
            Conv_Unit(1024, 1024, 0),               #15 conv
            Trans_Conv_Unit(1024, 512)               #1 transposed conv
                
        )

        self.decoder_2 = nn.Sequential(

            Conv_Unit(1024, 512, 0),                #16 conv
            Conv_Unit(512, 512, 0),                 #17 conv
            Conv_Unit(512, 512, 0),                 #18 conv
            Trans_Conv_Unit(512, 256)                #2 transposed conv

        )

        self.decoder_3 = nn.Sequential(

            Conv_Unit(512, 256, 0),                 #19 conv
            Conv_Unit(256, 256, 0),                 #20 conv
            Conv_Unit(256, 256, 0),                 #21 conv
            Trans_Conv_Unit(256, 128)                #3 transposed conv

        )

        self.decoder_4 = nn.Sequential(

            Conv_Unit(256, 128, 0),                 #22 conv
            Conv_Unit(128, 128, 0),                 #23 conv
            Conv_Unit(128, 128, 0),                 #24 conv
            Trans_Conv_Unit(128, 64)                 #4 transposed conv

        )

        self.decoder_5 = nn.Sequential(

            Conv_Unit(128, 64, 0),                  #25 conv
            Conv_Unit(64, 64, 0),                   #26 conv
            Conv_Unit(64, 64, 0),                   #27 conv
            Conv_Unit(64, 3)                        #28 conv [out]

        )

    def forward(self, L):
        first_encoded = self.encoder_1(L)
        second_encoded = self.encoder_2(first_encoded)
        third_encoded = self.encoder_3(second_encoded)
        fourth_encoded = self.encoder_4(third_encoded)

        bottleneck_first_decoded = self.bottleneck_decoder_1(fourth_encoded)                                
        bottleneck_first_decoded = torch.cat((bottleneck_first_decoded, fourth_encoded), dim = 1)           #skip connection 1

        second_decoded = self.decoder_2(bottleneck_first_decoded)
        second_decoded = torch.cat((second_decoded, third_encoded), dim = 1)                                #skip connection 2

        third_decoded = self.decoder_3(second_decoded)
        third_decoded = torch.cat((third_decoded, second_encoded), dim = 1)                                 #skip connection 3
        
        fourth_decoded = self.decoder_4(third_decoded)                                                      
        fourth_decoded = torch.cat((fourth_decoded, first_encoded), dim = 1)                                #skip connection 4

        out = self.decoder_5(fourth_decoded)

        return out
