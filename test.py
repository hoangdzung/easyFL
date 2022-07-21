import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing

def main():
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    model = server.get_model()
    data = server.get_client_data(0)
    print(len(data))

if __name__ == '__main__':
    main()




