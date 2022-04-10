import torch
import torch.nn as nn
import numpy as np
import utils

class CodedNet(nn.Module):

    def __init__(self):

        super(CodedNet, self).__init__()

        opt = utils.parse_arg()
        self.block_size = opt.blocksize
        self.channel = opt.channel

    # image compress
    def encode_CASSI(self, x, Mask):
        y = torch.mul(x, Mask)
        y = y.sum(3)
        return y

    # init x0 for deep unfolding
    def init_CASSI(self, y, Mask, channel):
        y1 = y.unsqueeze(3)
        y2 = y1.repeat([1, 1, 1, channel])
        x0 = torch.mul(y2, Mask)
        return x0

    # expand the size of the mask
    def expand_Cu(self, batch_size, T):
        Cu_input = np.zeros([self.block_size, self.block_size, self.channel])
        for ch in range(self.channel):
            Cu_input[:,:,ch] = np.roll(T, shift=-ch, axis=0)    #roll the mask
        Cu_input = np.expand_dims(Cu_input, axis=0)
        Cu_input = torch.Tensor(np.tile(Cu_input, [batch_size, 1, 1, 1]))
        return Cu_input

    def forward(self, x, T): 

        batch_size = x.size(0)      #batchsize
        Cu_input = self.expand_Cu(batch_size, T).cuda()    #expand mask_size
        y  = self.encode_CASSI(x, Cu_input)      #get measurements
        x0 = self.init_CASSI(y, Cu_input, self.channel)     #init x0

        return x0, Cu_input

def CodedModel():
    return CodedNet()
 