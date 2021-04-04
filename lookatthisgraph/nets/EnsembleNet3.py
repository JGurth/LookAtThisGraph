import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, knn_graph, DynamicEdgeConv
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dropout_adj
from torch.nn import BatchNorm1d
from lookatthisgraph.utils.LNN import LNN


class EnsembleNet3(torch.nn.Module):
    def __init__(self, n_features, n_labels, classification=False, width=128, conv_depth=3, lin_depth=5, aggr='max', point_depth=3):
        super(EnsembleNet3, self).__init__()
        self.classification = classification
        self.n_features = n_features
        self.n_labels = n_labels
        self.lin_depth=lin_depth
        self.conv_depth=conv_depth
        self.width=width
        self.point_depth=point_depth
        self.aggr=aggr
        n_intermediate = self.width

        self.conv1 = TAGConv(self.n_features, n_intermediate, 2)            
        self.convfkt=torch.nn.ModuleList([TAGConv(n_intermediate, n_intermediate, 2) for i in range(self.conv_depth-1)])
        
        self.point1 =  DynamicEdgeConv(LNN([2*n_features, n_intermediate, n_intermediate]), 3, self.aggr)
        self.pointfkt=torch.nn.ModuleList([DynamicEdgeConv(LNN([2*n_intermediate, n_intermediate]), 3, self.aggr) for i in range(self.point_depth-1)])
        
        
        n_intermediate2 = 2*self.conv_depth*n_intermediate+2*self.point_depth*n_intermediate
        self.dim2=n_intermediate2
        self.batchnorm1 = BatchNorm1d(n_intermediate2)                      
        self.linearfkt=torch.nn.ModuleList([torch.nn.Linear(n_intermediate2, n_intermediate2) for i in range(self.lin_depth)])
        self.drop = torch.nn.ModuleList([torch.nn.Dropout(.3) for i in range(self.lin_depth)])                                    
        self.out = torch.nn.Linear(n_intermediate2, self.n_labels)
        self.out2 = torch.nn.Linear(self.n_labels, self.n_labels)           


    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = knn_graph(x, 100, batch)                               #?
        edge_index, _ = dropout_adj(edge_index, p=0.3)                      #?
        batch = data.batch

        x = F.leaky_relu(self.conv1(x, edge_index))
        x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)       
        convlist=[x1]   #dim = n_intermediate*2
        
        for f in range(self.conv_depth-1):
            x = F.leaky_relu(self.convfkt[f](x, edge_index))
            xi = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)
            convlist.append(xi)
        
        x = torch.cat(convlist, dim=1)   #dim = n_intermediate*2*conv_depth
        
        y=data.x
        y=self.point1(y, batch)  #dim=n_intermediate
        pointlist=[y]
        for f in range(self.point_depth-1):
            y=self.pointfkt[f](y, batch)
            ####
            pointlist.append(y)
        
        y=torch.cat(pointlist, dim=1) #dim=n_intermediate*point_depth
        y = torch.cat([gap(y, batch), gmp(y, batch)], dim=1)

        
        x=torch.cat([x, y], dim=1)

        x = self.batchnorm1(x)
        for g in range(self.lin_depth):
            x=F.leaky_relu(self.linearfkt[g](x))
            if (g-1)%3==0 and self.lin_depth-1>g:  #g=1,4,7,... u. noch mind. zwei weitere Layers
                x = self.drop[g](x)


        x = self.out(x)
        if self.classification:
            x = torch.sigmoid(x)
        x = x.view(-1)

        return x
