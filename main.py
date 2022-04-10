# utility functions
#train for paper
import train_for_paper.model as model_paper # model implementation
import train_for_paper.trainer as trainer_paper # training functions
#train for cave
import train_for_cave.model as model_cave
import train_for_cave.trainer as trainer_cave
import utils # other utilities
import optimizer # optimization functions

# public libraries
import torch
import logging
import numpy as np
import time
import os

def main():

    # logging configuration
    logging.basicConfig(level = logging.INFO,
        format = "[%(asctime)s]: %(message)s"
    )
    
    # parse command line input
    opt = utils.parse_arg()

    # Set GPU
    opt.cuda = opt.gpuid>=0
    if opt.cuda:
        torch.cuda.set_device(opt.gpuid)
    else:
        raise NotImplementedError

    # record the current time
    opt.save_dir += time.asctime(time.localtime(time.time()))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # initialize the model or loading the pre-train model
    if opt.mode == 'paper':
        model_train = model_paper.prepare_model(opt)
    elif opt.mode == 'cave':
        model_train = model_cave.prepare_model(opt)
    else:
        raise NotImplementedError

    if opt.pretrained:
        model_train.load_state_dict(torch.load(opt.pretrained_path,  map_location='cuda:0') )
    
    # configurate the optimizer and learning rate scheduler
    optim, sche = optimizer.prepare_optim(model_train, opt)

    # train the model
    if opt.mode == 'paper':
        model_train = trainer_paper.train(model_train, optim, sche, opt)
    elif opt.mode == 'cave':
        model_train = trainer_cave.train(model_train, optim, sche, opt)
    else:
        raise NotImplementedError

    # save the final trained model
    utils.save_model(model_train, opt)
        
    return 

if __name__ == '__main__':
    main()