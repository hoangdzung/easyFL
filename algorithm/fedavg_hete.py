from .fedbase import BasicServer, BasicClient
import numpy as np 

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def sample(self):
        selected_clients = super().sample()
        selected_clients = [idx for idx in selected_clients if self.clients[idx].model_type ==0]
        return selected_clients

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.model_type = 0 if np.random.rand() < option['small_machine_rate'] else 1
