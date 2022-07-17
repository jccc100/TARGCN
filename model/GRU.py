import torch
import torch.nn as nn
from model.EmbGCN import EmbGCN as GCN              # TARGCN
# from model.EmbGCN import EmbGCN_linear as GCN     # TARGCN-linear
# from model.EmbGCN import EmbGCN_SA as GCN         # TARGCN-SA

class GRU(nn.Module):
    def __init__(self, node_num, dim_in, dim_out,adj, cheb_k, embed_dim):
        super(GRU, self).__init__()
        self.adj=adj
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in+self.hidden_dim, 2*dim_out, self.adj, cheb_k, embed_dim)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, self.adj, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)