from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import numpy as np 
import os
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn import metrics
import copy 

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.seed = option['seed']

    def compare_model(self, encoded_inputs):
        n_selected_clients = len(self.selected_clients)
        labels_to_clients = defaultdict(list)

        X = []
        for i, (client, encoded_input) in enumerate(zip(self.selected_clients, encoded_inputs)):
            labels_to_clients[tuple(sorted(self.clients[client].all_labels))].append(i)
            X.append(encoded_input.detach().cpu().numpy())
        p = np.zeros(len(X))
        y = np.zeros(len(X))
        for i, client_ids in enumerate(labels_to_clients.values()):
            y[client_ids] = i 
            p[client_ids] = 1/len(client_ids)
        kmeans = KMeans(n_clusters=self.num_groups, random_state=self.seed).fit(X)   
        y_pred = kmeans.labels_
        print(metrics.homogeneity_score(y, y_pred), metrics.completeness_score(y, y_pred))
        return p.tolist()

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        return models, train_losses

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
        models, train_losses, encoded_inputs = self.communicate(self.selected_clients, pool)
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        p = self.compare_model(encoded_inputs)
        # self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])
        self.model = self.aggregate(models, p = p)
        # self.aggregate(models)
        
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.all_labels = self.train_data.all_labels if self.train_data else [] 

    def pack(self, model, loss, encoded_input):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "train_loss": loss,
            "encoded_input": encoded_input
        }

    def train(self, model, device):
        momentum = 0.99
        encoded_inputs = None
        model = model.to(device)
        model.train()
                        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, encoded_input = self.get_loss(model, batch_data, device)
                if encoded_inputs is None:
                    encoded_inputs = encoded_input.mean(0)
                else:
                    encoded_inputs = momentum * encoded_inputs + (1 - momentum) * encoded_input.mean(0)
                    
                loss.backward()
                optimizer.step()
        return encoded_inputs
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, data, device):
        tdata = self.data_to_device(data, device)    
        output_s, representation_s = model.pred_and_rep(tdata[0])                 
        loss = self.lossfunc(output_s, tdata[1])
        return loss, representation_s

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
        encoded_inputs = self.train(model, device)
        cpkg = self.pack(model, loss, encoded_inputs)
        return cpkg