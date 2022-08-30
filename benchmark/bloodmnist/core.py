from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
import medmnist
from medmnist.dataset import *

class WrapPathMNIST(BloodMNIST):
    def __init__(self, **kwargs):
        super(WrapPathMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapPathMNIST, self).__getitem__(idx)
        return image, int(label[0])

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, num_groups=3, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='bloodmnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      num_groups=num_groups,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/bloodmnist/data',
                                      )
        self.num_classes = 11
        self.save_data = self.XYData_to_json
   

    def load_data(self):

        self.train_data = WrapPathMNIST(root=self.rawdata_path, split='train', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]))
        self.test_data = WrapPathMNIST(root=self.rawdata_path, split='test', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]))

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

