from .fedbase import BasicServer, BasicClient

class Server(BasicServer):
    def __init__(self, option, model, clients, valid_data=None, test_data = None):
        super(Server, self).__init__(option, model, clients, valid_data, test_data)

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


