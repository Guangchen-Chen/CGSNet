from torchviz import make_dot
import torch
from tensorboardX import SummaryWriter

from encoder import Deeplabv3plusEncoder, PSPNetEncoder, DeepLabV2Encoder
from decoders import *



def model_select(modelName = None, backbone='resnet50', nclass=6):
    if not modelName:
        return
    model = modelName(backbone, nclass)
    return model


def dmodel_select(Decoder, nclss, size, conv_in_ch):
    if not Decoder:
        return

    if Decoder == MainDecoder:
        upscale = UPSAMPLE
        dmodel = Decoder(upscale, conv_in_ch, nclss)
        return dmodel
    else:
        upscale = UPSAMPLE
        dmodel = Decoder(upscale, conv_in_ch, nclss)
        return dmodel

def show_model(input, mask, backbone, nclass, modelName=None, Decoder=None, isGPU=False, tensorboardShow=False):
    if not modelName:
        return

    if isGPU:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    emodel = model_select(modelName, backbone, nclass)
    emodel = emodel.to(device)
    input.to(device)
    print('Encoder输入：'+ str(input.shape))
    out = emodel(input)
    print('Eecoder输出：' + str(out.shape))
    print('Dncoder输入：' + str(out.shape))
    if Decoder == MainDecoder:
        dmodel = dmodel_select(Decoder, nclass, size=input.size(), conv_in_ch = out.size()[1])
        dmodel = dmodel.to(device)
        output = dmodel(out)
    else:
        dmodel = dmodel_select(Decoder, nclass, size=input.size(), conv_in_ch=out.size()[1])
        dmodel = dmodel.to(device)
        print('mask输入：' + str(mask.shape))
        output = dmodel(out, mask)

    print('Decoder输出：' + str(output.shape))

    if tensorboardShow:
        writer = SummaryWriter("./log", comment="sample_model_visualization")
        writer.add_graph(emodel, (input,))
        writer.add_graph(dmodel, (out,))


UPSAMPLE = 8

if __name__ == '__main__':
    '''
    编码器选择：
    Deeplabv3plusEncoder (up=4)
    PSPNetEncoder (up=8)
    DeepLabV2Encoder
    '''

    '''
    解码器选择：
    MainDecoder
    CutOutDecoder
    ContextMaskingDecoder
    ObjectMaskingDecoder
    FeatureDropDecoder
    FeatureNoiseDecoder
    DropOutDecoder
    '''

    input = torch.rand((4, 3, 320, 320))
    mask = torch.rand((4, 3, 320, 320))
    backbone = 'resnet50'
    nclass = 6
    modelName =  DeepLabV2Encoder
    Decoder = MainDecoder
    isGPU = False
    tensorboardShow = False

    show_model(input, mask, backbone, nclass, modelName, Decoder, isGPU, tensorboardShow)


