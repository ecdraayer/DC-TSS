import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size1=32, size2=5):
        return input.view(input.size(0), size1, size2)
    
class Autoencoder(nn.Module):
    
    def __init__(self,input_shape, k_sizes=[15,12,7,4], strides=[2,2,2,2]):
        super().__init__()
        modules = []
        
        
        for i in range(len(k_sizes)):
            modules.append(nn.Conv1d(input_shape[1] , input_shape[1], k_sizes[i], stride=strides[i]))
            if i != len(k_sizes)-1:
                modules.append(nn.LeakyReLU(0.1)) 
        self.encoder = nn.Sequential(*modules)
        
#         self.encoder = nn.Sequential(
#             nn.Conv1d(input_shape[1], input_shape[1], 15, stride=2), # increase chan
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(input_shape[1], input_shape[1], 12, stride=2),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(input_shape[1], input_shape[1], 7, stride=2),
#             nn.LeakyReLU(0.1),
#             nn.Conv1d(input_shape[1], input_shape[1], 4, stride=2),

#         )
        modules = []
        modules.append(nn.Upsample(size=input_shape[-1], mode='nearest'))
        for i in range(len(k_sizes)):
            modules.append(nn.LeakyReLU(0.1))
            modules.append(nn.ConvTranspose1d(input_shape[1], input_shape[1], k_sizes[i], stride=strides[i]))
        
        modules.reverse()
        
        self.decoder = nn.Sequential(*modules)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(input_shape[1], input_shape[1], 4, stride=2),
#             nn.LeakyReLU(0.1),
#             nn.ConvTranspose1d(input_shape[1], input_shape[1], 7, stride=2),
#             nn.LeakyReLU(0.1),
#             nn.ConvTranspose1d(input_shape[1], input_shape[1], 12, stride=2),
#             nn.LeakyReLU(0.1),
#             nn.ConvTranspose1d(input_shape[1], input_shape[1], 15 , stride=2),
#             nn.LeakyReLU(0.1),
#             nn.Upsample(size=input_shape[-1], mode='nearest')
#         )
        
        
    def encode(self, x):
        h = self.encoder(x)
        return h
    
    def decode(self, z):
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z
    
    def get_encoder(self):
        return self.encoder
    
    def init_weights(self):
        def func(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.00)

        self.encoder.apply(func)
        self.decoder.apply(func)
        
class clustering(nn.Module):

    def __init__(self, n_clusters:int, alpha: float = 1.0, cluster_centers = None) -> None:
        super(clustering, self).__init__()

        self.n_clusters = n_clusters
        self.alpha = alpha

#         if cluster_centers is None:
#             initial_cluster_centers = torch.zeros(self.n_clusters, 40, dtype=torch.float32)
#             nn.init.xavier_uniform_(initial_cluster_centers)
#         else:
#             initial_cluster_centers = cluster_centers
#         self.clustcenters = nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        """ student t-distribution, as same as used in t-SNE algorithm.
            inputs: the variable containing data, shape=(n_samples, n_features)
            output: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        inputs = torch.flatten(inputs, start_dim=1)
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, axis=1) - self.clustcenters), axis=2) / self.alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
class DEC(nn.Module):
    def __init__(self, n_clusters, input_shape, k_sizes=[15,12,7,4], strides=[2,2,2,2]):
        super(DEC, self).__init__()
        
        self.AE = Autoencoder(input_shape, k_sizes, strides)
        self.clustlayer = clustering(n_clusters)

        self.model = nn.Sequential(
            self.AE.encoder,
            self.clustlayer)    
        
    def forward(self, inputs):
        X = self.model(inputs)
        return X

    
def pretraining(
    model:torch.nn.Module, 
    dbgenerator:object, 
    batch_size: int=256,
    epochs: int=10,
    savepath: str = './save/models/', 
    device = 'cuda:0',
    savemodel: bool=True,
    lr: float=1.5e-4,
    ):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.AE.parameters(), lr)
    data_loader = dbgenerator
    for epoch in range(epochs):

        model.AE.train()  # Set model to training mode

        for ts in data_loader:
           
            ts = ts.type(torch.cuda.FloatTensor).to(device)
            optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(True):
                # forward
                outputs = model.AE(ts)
                loss = criterion(outputs, ts)

                # backward
                loss.backward()
                optimizer.step()
            
        print(f'epoch {epoch+1},loss = {loss:.8f}')

    #if savemodel:
    #   if not os.path.exists(savepath):
    #        os.mkdir(savepath)
    #    torch.save(model.AE.state_dict(), os.path.join(savepath,'ae_weights.pth'))

    
def refine_clusters(n_clusters, data_loader, deep_cluster_model, device, epochs2, batch_size, lr, momentum, update_interval):
    
    deep_cluster_model.train()

    with torch.no_grad():
        print('Initializing cluster centers with k-means. number of clusters %s' % n_clusters)

        clustering_output = []
        for ts in data_loader:
            recon = deep_cluster_model.AE.encode((ts.float().to(device)))
            clustering_output.append( recon.cpu().detach().numpy() ) 

        
        #print(len(clustering_output))
        clustering_output = [item for sublist in clustering_output for item in sublist]
        clustering_output = np.asarray(clustering_output)
        clustering_output_f = []
        for i,co in enumerate(clustering_output):
            clustering_output[i].flatten()
            clustering_output_f.append(clustering_output[i].flatten())
        clustering_output_f = np.asarray(clustering_output_f)
        
#         initial_cluster_centers = torch.zeros(n_clusters, len(clustering_output_f[0]), dtype=torch.float32)
#         nn.init.xavier_uniform_(initial_cluster_centers)
#         initial_cluster_centers = initial_cluster_centers.to(device)
#         deep_cluster_model.clustlayer.clustcenters = torch.nn.Parameter(initial_cluster_centers)
        
#         kmeans = AgglomerativeClustering(n_clusters=n_clusters)#KMeans(n_clusters=n_clusters, n_init=200)
#         y_pred_last = kmeans.fit_predict(clustering_output_f)
        
#         clf = NearestCentroid()
#         clf.fit(clustering_output_f, y_pred_last)
        
#         clustcenters = torch.tensor(clf.centroids_, dtype=torch.float, requires_grad=True)
#         clustcenters = clustcenters.to(device)
#         #self.clustcenters = nn.Parameter(initial_cluster_centers)
#         deep_cluster_model.clustlayer.clustcenters = torch.nn.Parameter(clustcenters)
#         deep_cluster_model.state_dict()["clustlayer.clustcenters"].copy_(clustcenters)
        kmeans = KMeans(n_clusters=n_clusters, n_init=150)#KMeans(n_clusters=n_clusters, n_init=200)#
        y_pred_last = kmeans.fit_predict(clustering_output_f)
        #clf = NearestCentroid()
        #clf.fit(clustering_output_f, y_pred_last)
        
        clustcenters = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
        clustcenters = clustcenters.to(device)
        #self.clustcenters = nn.Parameter(initial_cluster_centers)
        deep_cluster_model.clustlayer.clustcenters = torch.nn.Parameter(clustcenters)
        deep_cluster_model.state_dict()["clustlayer.clustcenters"].copy_(clustcenters)

    
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(deep_cluster_model.model.parameters(), lr=lr, momentum=momentum)

    
    index = 0
    loss = 0
    count = 0
    X = data_loader.dataset.float().to(device)
    index_array = np.arange(X.shape[0])
    print(X.shape)
    update_interval = np.ceil(X.shape[0]/batch_size) * update_interval
    iterations = epochs2 * int(update_interval)
    print(update_interval)
    for i in range(iterations):
        # instead of in every iteration, we update the auxiliary target distribution in every update_inteval
        if i % update_interval == 0:
            index_array = np.arange(X.shape[0])
            with torch.no_grad():
                q = deep_cluster_model(X)
                p = deep_cluster_model.clustlayer.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)
                if i != 0:
                    print('Epoch %d: ' % (i // update_interval), ' loss=', np.round(loss/count, 5))

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            #idx = index_array[index * batch_size: min((index + 1) * batch_size, X.shape[0])]
            idx = np.random.choice(index_array, min(batch_size,len(index_array)),replace=False)
            trainx = X[idx]
            trainx = trainx.to(device)
            trainy = p[idx]
            trainy = trainy.to(device)

            outputs = deep_cluster_model(trainx)
            value_idx = [i for i,val in enumerate(index_array) if val in idx]
            index_array = np.delete(index_array, value_idx)
            train_loss = criterion(outputs.log(), trainy)

            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            count +=1
    y_pred_last = y_pred.detach().clone().cpu().numpy()
    return y_pred_last

#         if iterations % np.ceil(X.shape[0]/batch_size) == 0:
#             q = deep_cluster_model(X)
#             y_pred = q.argmax(1)
#             print('Epoch %d: ' % (iterations / np.ceil(X.shape[0]/batch_size)), ' loss=', np.round(loss/count, 5))

#             # check stop criterion, when less than tol%of points change cluster assignment between two consecutive epochs.
#             delta_label = np.sum(y_pred_last!= y_pred.clone().detach().cpu().numpy()) / y_pred.shape[0]

#             y_pred_last = y_pred.detach().clone().cpu().numpy()