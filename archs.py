import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['UNet', 'UNetNBN' , 'NestedUNet','PyramidUNet','PyramidNestedUNet','FCNN','FCNN2','FCNNhub','UNetBnout']

class FCNNhub(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes
        nb_filter = [1]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv0_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(self.n_classes)

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return [x1,x2]

    # def forward(self, input):
    #     x0_0 = self.pool(self.conv0_0(input))
    #     x0_1 = self.conv0_1(self.ajust_padding(input,self.up(x0_0))[1])
    #     output = self.final(x0_1)
    #     return output
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_1 = self.conv0_1(x0_0)
        output = self.final(x0_1)
        output=self.bn_out(output)
        return output

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,use_bn=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)

        return out

# no batch normalization UNet
class UNetNBN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0],use_bn=False)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],use_bn=False)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],use_bn=False)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],use_bn=False)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],use_bn=False)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],use_bn=False)
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],use_bn=False)
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],use_bn=False)
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],use_bn=False)

        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return [x1,x2]

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat(self.ajust_padding(x3_0, self.up(x4_0)), 1))
        x2_2 = self.conv2_2(torch.cat(self.ajust_padding(x2_0, self.up(x3_1)), 1))
        x1_3 = self.conv1_3(torch.cat(self.ajust_padding(x1_0, self.up(x2_2)), 1))
        x0_4 = self.conv0_4(torch.cat(self.ajust_padding(x0_0, self.up(x1_3)), 1))

        output = self.final(x0_4)
        return output

class FCNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes
        nb_filter = [32]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv0_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return [x1,x2]

    # def forward(self, input):
    #     x0_0 = self.pool(self.conv0_0(input))
    #     x0_1 = self.conv0_1(self.ajust_padding(input,self.up(x0_0))[1])
    #     output = self.final(x0_1)
    #     return output
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_1 = self.conv0_1(x0_0)
        output = self.final(x0_1)
        return output

class FCNN2(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes
        nb_filter = [32]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv0_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return [x1,x2]

    def forward(self, input):
        x0_0 = self.pool(self.conv0_0(input))
        x0_1 = self.conv0_1(self.ajust_padding(input,self.up(x0_0))[1])
        output = self.final(x0_1)
        return output
    # def forward(self, input):
    #     x0_0 = self.conv0_0(input)
    #     x0_1 = self.conv0_1(x0_0)
    #     output = self.final(x0_1)
    #     return output

class UNetBnout(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(self.n_classes)

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return [x1,x2]
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat(self.ajust_padding(x3_0, self.up(x4_0)), 1))
        x2_2 = self.conv2_2(torch.cat(self.ajust_padding(x2_0, self.up(x3_1)), 1))
        x1_3 = self.conv1_3(torch.cat(self.ajust_padding(x1_0, self.up(x2_2)), 1))
        x0_4 = self.conv0_4(torch.cat(self.ajust_padding(x0_0, self.up(x1_3)), 1))

        output = self.final(x0_4)
        output=self.bn_out(output)
        return output


class UNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return [x1,x2]
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat(self.ajust_padding(x3_0, self.up(x4_0)), 1))
        x2_2 = self.conv2_2(torch.cat(self.ajust_padding(x2_0, self.up(x3_1)), 1))
        x1_3 = self.conv1_3(torch.cat(self.ajust_padding(x1_0, self.up(x2_2)), 1))
        x0_4 = self.conv0_4(torch.cat(self.ajust_padding(x0_0, self.up(x1_3)), 1))

        output = self.final(x0_4)
        return output

class PyramidUNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0]+self.n_channels, nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1]+self.n_channels, nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2]+self.n_channels, nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3]+self.n_channels, nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return [x1,x2]
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        input_0 = self.pool(input)
        x1_0 = self.conv1_0(torch.cat((self.pool(x0_0),input_0),1))
        input_1 = self.pool(input_0)
        x2_0 = self.conv2_0(torch.cat((self.pool(x1_0),input_1),1))
        input_2 = self.pool(input_1)
        x3_0 = self.conv3_0(torch.cat((self.pool(x2_0),input_2),1))
        input_3 = self.pool(input_2)
        x4_0 = self.conv4_0(torch.cat((self.pool(x3_0),input_3),1))

        x3_1 = self.conv3_1(torch.cat(self.ajust_padding(x3_0, self.up(x4_0)), 1))
        x2_2 = self.conv2_2(torch.cat(self.ajust_padding(x2_0, self.up(x3_1)), 1))
        x1_3 = self.conv1_3(torch.cat(self.ajust_padding(x1_0, self.up(x2_2)), 1))
        x0_4 = self.conv0_4(torch.cat(self.ajust_padding(x0_0, self.up(x1_3)), 1))

        output = self.final(x0_4)
        return output

class PyramidNestedUNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes
        self.deep_supervision = args.deep_supervision

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0]+self.n_channels, nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1]+self.n_channels, nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2]+self.n_channels, nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3]+self.n_channels, nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def pair_ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return [x1,x2]

    def ajust_padding(self,map_list):
        ajust_list = []
        ajust_list.append(map_list[0])
        for i in range(1,len(map_list)):
            tmp = self.pair_ajust_padding(map_list[0],map_list[i])
            ajust_list.append(tmp[1])
        return ajust_list

    def forward(self, input):
        input_0 = self.pool(input)
        input_1 = self.pool(input_0)
        input_2 = self.pool(input_1)
        input_3 = self.pool(input_2)

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(torch.cat((self.pool(x0_0),input_0),1))
        x0_1 = self.conv0_1(torch.cat(self.ajust_padding([x0_0, self.up(x1_0)]), 1))

        x2_0 = self.conv2_0(torch.cat((self.pool(x1_0),input_1),1))
        x1_1 = self.conv1_1(torch.cat(self.ajust_padding([x1_0, self.up(x2_0)]), 1))
        x0_2 = self.conv0_2(torch.cat(self.ajust_padding([x0_0, x0_1, self.up(x1_1)]), 1))

        x3_0 = self.conv3_0(torch.cat((self.pool(x2_0),input_2),1))
        x2_1 = self.conv2_1(torch.cat(self.ajust_padding([x2_0, self.up(x3_0)]), 1))
        x1_2 = self.conv1_2(torch.cat(self.ajust_padding([x1_0, x1_1, self.up(x2_1)]), 1))
        x0_3 = self.conv0_3(torch.cat(self.ajust_padding([x0_0, x0_1, x0_2, self.up(x1_2)]), 1))

        x4_0 = self.conv4_0(torch.cat((self.pool(x3_0),input_3),1))
        x3_1 = self.conv3_1(torch.cat(self.ajust_padding([x3_0, self.up(x4_0)]), 1))
        x2_2 = self.conv2_2(torch.cat(self.ajust_padding([x2_0, x2_1, self.up(x3_1)]), 1))
        x1_3 = self.conv1_3(torch.cat(self.ajust_padding([x1_0, x1_1, x1_2, self.up(x2_2)]), 1))
        x0_4 = self.conv0_4(torch.cat(self.ajust_padding([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)]), 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class NestedUNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.n_channels = args.input_channels
        self.n_classes = args.num_classes
        self.deep_supervision = args.deep_supervision

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def pair_ajust_padding(self,x1,x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return [x1,x2]

    def ajust_padding(self,map_list):
        ajust_list = []
        ajust_list.append(map_list[0])
        for i in range(1,len(map_list)):
            tmp = self.pair_ajust_padding(map_list[0],map_list[i])
            ajust_list.append(tmp[1])
        return ajust_list

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat(self.ajust_padding([x0_0, self.up(x1_0)]), 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat(self.ajust_padding([x1_0, self.up(x2_0)]), 1))
        x0_2 = self.conv0_2(torch.cat(self.ajust_padding([x0_0, x0_1, self.up(x1_1)]), 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat(self.ajust_padding([x2_0, self.up(x3_0)]), 1))
        x1_2 = self.conv1_2(torch.cat(self.ajust_padding([x1_0, x1_1, self.up(x2_1)]), 1))
        x0_3 = self.conv0_3(torch.cat(self.ajust_padding([x0_0, x0_1, x0_2, self.up(x1_2)]), 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat(self.ajust_padding([x3_0, self.up(x4_0)]), 1))
        x2_2 = self.conv2_2(torch.cat(self.ajust_padding([x2_0, x2_1, self.up(x3_1)]), 1))
        x1_3 = self.conv1_3(torch.cat(self.ajust_padding([x1_0, x1_1, x1_2, self.up(x2_2)]), 1))
        x0_4 = self.conv0_4(torch.cat(self.ajust_padding([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)]), 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
