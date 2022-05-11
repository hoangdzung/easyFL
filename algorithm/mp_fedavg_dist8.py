from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import numpy as np 
import os
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn import metrics

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.num_groups = option['num_groups']
        self.seed = option['seed']

    def compare_model(self, models):
        n_selected_clients = len(self.selected_clients)
        labels_to_clients = defaultdict(list)

        X = []
        for i, (client, model) in enumerate(zip(self.selected_clients, models)):
            labels_to_clients[tuple(sorted(client[i].all_labels))].append(i)
            X.append(model.fc1.weight.data.flatten().detach().cpu().numpy())
        
        y = np.zeros(len(X))
        for i, client_ids in enumerate(labels_to_clients.values()):
            y[client_ids] = i 

        kmeans = KMeans(n_clusters=self.num_groups, random_state=self.seed).fit(X)   
        y_pred = kmeans.labels_
        print(metrics.homogeneity_score(y, y_pred), metrics.completeness_score(y, y_pred))
    
    def iterate(self, t, pool):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients, pool)
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])
        self.compare_model(models)
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.all_labels = self.train_data.all_labels if self.train_data else [] 

    def mask_weight(self, model):
        
        missing_labels = [i for i in range(model.fc2.weight.shape[0]) if i not in self.all_labels ]
        model.fc2.weight.data[missing_labels,:] = 0
                
    def reply(self, svr_pkg, device):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model, device)
        self.train(model, device)
        self.mask_weight(model)
        cpkg = self.pack(model, loss)
        return cpkg

