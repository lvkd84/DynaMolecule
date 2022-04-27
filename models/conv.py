import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag

#https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
class GCNConv(MessagePassing):
    def __init__(self, 
                 emb_dim=256):
        '''
            emb_dim (int): node/edge/model embedding dimensionality
        '''
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_linear = torch.nn.Linear(emb_dim, 1)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    # def reset_parameters(self):
    #     pass

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index

        edge_weight = torch.squeeze(self.edge_linear(edge_attr),dim=-1)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_weight = edge_weight, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

#https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
class GINEConv(MessagePassing):
    def __init__(self, 
                 emb_dim=256,
                 train_ep=False):
        '''
            emb_dim (int): node/edge/model embedding dimensionality
        '''

        super(GINEConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        if train_ep:
            self.eps = torch.nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.]))

    # def reset_parameters(self):
    #     pass

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps)*x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out



# class GATv2Conv(MessagePassing):

#     _alpha: OptTensor

#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#         dropout: float = 0.0,
#         add_self_loops: bool = True,
#         edge_dim: Optional[int] = None,
#         fill_value: Union[float, Tensor, str] = 'mean',
#         bias: bool = True,
#         share_weights: bool = False,
#     ):
#         super().__init__(node_dim=0)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value
#         self.share_weights = share_weights

#         if isinstance(in_channels, int):
#             self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
#                                 weight_initializer='glorot')
#             if share_weights:
#                 self.lin_r = self.lin_l
#             else:
#                 self.lin_r = Linear(in_channels, heads * out_channels,
#                                     bias=bias, weight_initializer='glorot')
#         else:
#             self.lin_l = Linear(in_channels[0], heads * out_channels,
#                                 bias=bias, weight_initializer='glorot')
#             if share_weights:
#                 self.lin_r = self.lin_l
#             else:
#                 self.lin_r = Linear(in_channels[1], heads * out_channels,
#                                     bias=bias, weight_initializer='glorot')

#         self.att = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#         else:
#             self.lin_edge = None

#         if bias and concat:
#             self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self._alpha = None

#         self.reset_parameters()


#     def reset_parameters(self):
#         self.lin_l.reset_parameters()
#         self.lin_r.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att)
#         zeros(self.bias)


#     def forward(self, x, edge_index, edge_attr = None, return_attention_weights = None):

#         H, C = self.heads, self.out_channels

#         x_l = None
#         x_r = None
#         if isinstance(x, torch.Tensor):
#             assert x.dim() == 2
#             x_l = self.lin_l(x).view(-1, H, C)
#             if self.share_weights:
#                 x_r = x_l
#             else:
#                 x_r = self.lin_r(x).view(-1, H, C)
#         else:
#             x_l, x_r = x[0], x[1]
#             assert x[0].dim() == 2
#             x_l = self.lin_l(x_l).view(-1, H, C)
#             if x_r is not None:
#                 x_r = self.lin_r(x_r).view(-1, H, C)

#         assert x_l is not None
#         assert x_r is not None

#         if self.add_self_loops:
#             if isinstance(edge_index, torch.Tensor):
#                 num_nodes = x_l.size(0)
#                 if x_r is not None:
#                     num_nodes = min(num_nodes, x_r.size(0))
#                 edge_index, edge_attr = remove_self_loops(
#                     edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # propagate_type: (x: PairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
#                              size=None)

#         alpha = self._alpha
#         self._alpha = None

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias

#         if isinstance(return_attention_weights, bool):
#             assert alpha is not None
#             if isinstance(edge_index, torch.Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out


#     def message(self, x_j, x_i, edge_attr, index, ptr=None, size_i=None):
#         x = x_i + x_j

#         if edge_attr is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             assert self.lin_edge is not None
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             x += edge_attr

#         x = F.leaky_relu(x, self.negative_slope)
#         alpha = (x * self.att).sum(dim=-1)
#         alpha = softmax(alpha, index, ptr, size_i)
#         self._alpha = alpha
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return x_j * alpha.unsqueeze(-1)