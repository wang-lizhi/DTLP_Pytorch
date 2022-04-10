import torch.nn as nn
import train_for_paper.CodedModel as CodedModel
import train_for_paper.ReconModel as ReconModel

class DTLP(nn.Module):
    def __init__(self, codednet, reconnet):

        super(DTLP, self).__init__()
        self.codednet = codednet
        self.reconnet = reconnet
        
    def forward(self, x):

        x, Cu = self.codednet(x)
        Output_hsi = self.reconnet(x,Cu)

        return Output_hsi

def prepare_model(opt):
    # DTLP consists of two parts:
    #1. the compression
    #2. the HSI reconstruction
    codedmodel = CodedModel.CodedModel()
    reconmodel = ReconModel.ReconModel()   
    model = DTLP(codedmodel, reconmodel)     
    
    if opt.cuda:
        model = model.cuda()
    else:
        raise NotImplementedError

    return model