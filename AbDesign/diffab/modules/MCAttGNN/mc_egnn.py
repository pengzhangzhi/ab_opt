#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    Most codes are copied from https://github.com/vgsatorras/egnn, which is the official implementation of
    the paper:
        E(n) Equivariant Graph Neural Networks
        Victor Garcia Satorras, Emiel Hogeboom, Max Welling
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_softmax
from diffab.modules.common.geometry import quaternion_1ijk_to_rotation_matrix

from diffab.modules.common.so3 import so3vec_to_rotation


def exists(x):
    return x is not None
class MC_E_GCL(nn.Module):
    """
    Multi-Channel E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False,
                 dropout=0.1,time_emb_dim = None,):
        super(MC_E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 2*(n_channel)**2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        
        # rot_layer = nn.Linear(hidden_nf, n_channel, bias=False)
        # torch.nn.init.xavier_uniform_(rot_layer.weight, gain=0.001)
        # rot_mlp = []
        # rot_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        # rot_mlp.append(act_fn)
        # rot_mlp.append(rot_layer)
        # if self.tanh:
        #     rot_mlp.append(nn.Tanh())
        # self.rot_mlp = nn.Sequential(*rot_mlp)
        
        # self.to_rot = nn.Linear(n_channel * 3, 3, bias=True)
        # with torch.no_grad():
        #     self.to_rot.weight.fill_(0.0)
            
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, output_nf * 2)
        ) if exists(time_emb_dim) else None
        
    def edge_model(self, source, target, radial, edge_attr):
        '''
        :param source: [n_edge, input_size]
        :param target: [n_edge, input_size]
        :param radial: [n_edge, n_channel, n_channel]
        :param edge_attr: [n_edge, edge_dim]
        '''
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, n_channel ^ 2]

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        :param x: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param edge_attr: [n_edge, hidden_size], refers to message from i to j
        :param node_attr: [bs * n_node, node_dim]
        '''
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [bs * n_node, hidden_size]
        # print(f'agg1, {torch.isnan(agg).sum()}', )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [bs * n_node, input_size + hidden_size]
        # print(f'agg, {torch.isnan(agg).sum()}', )
        out = self.node_mlp(agg)  # [bs * n_node, output_size]
        # print(f'out, {torch.isnan(out).sum()}', )
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        '''
        coord: [bs * n_node, n_channel, d]
        edge_index: list of [n_edge], [n_edge]
        coord_diff: [n_edge, n_channel, d]
        edge_feat: [n_edge, hidden_size]
        '''
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)  # [n_edge, n_channel, d]

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))  # [bs * n_node, n_channel, d]
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord
    
    # def rot_model(self, rot, edge_index, coord_diff, edge_feat):
    #     '''
    #     rot: [bs * n_node, 3, 3]
    #     edge_index: list of [n_edge], [n_edge]
    #     coord_diff: [n_edge, n_channel, d]
    #     edge_feat: [n_edge, hidden_size]
    #     '''
    #     row, col = edge_index
    #     msg = coord_diff * self.rot_mlp(edge_feat).unsqueeze(-1)  # [n_edge, n_channel, d]

    #     if self.coords_agg == 'sum':
    #         agg = unsorted_segment_sum(msg, row, num_segments=rot.size(0))
    #     elif self.coords_agg == 'mean':
    #         agg = unsorted_segment_mean(msg, row, num_segments=rot.size(0))  # [bs * n_node, n_channel, d]
    #     else:
    #         raise Exception('Wrong coords_agg parameter' % self.coords_agg)
    #     n_nodes = rot.shape[0]
    #     agg = agg.reshape(n_nodes, -1) # [bs * n_node, n_channel * d]
    #     rot_update_vec = self.to_rot(agg) # [bs * n_node, 3]
    #     rot = rot @ quaternion_1ijk_to_rotation_matrix(rot_update_vec)
    #     # quat = rot_to_quat(rot) # [bs * n_node, 4]
    #     # rot = compose_q_update_vec(quat, rot_update_vec) # [bs * n_node, 3, 3]
    #     return rot

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None,time_emb=None):
        '''
        h: [bs * n_node, hidden_size]
        edge_index: list of [n_row] and [n_col] where n_row == n_col (with no cutoff, n_row == bs * n_node * (n_node - 1))
        coord: [bs * n_node, n_channel, d]
        '''
        row, col = edge_index
        radial, dist, coord_diff = coord2radial(edge_index, coord)
        # concatenate radial and dist at the last two dimensions
        radial = torch.cat([radial, dist], dim=-1) # [n_edge, n_channel, 2 * n_channel]
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [n_edge, hidden_size]
        # print(f'edge_feat, {torch.isnan(edge_feat).sum()}', )
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)    # [bs * n_node, n_channel, d]
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            scale, shift = time_emb.chunk(2, dim = 1)
            h = h * (scale + 1) + shift

        return h, coord


class MC_Att_L(nn.Module):
    """
    Multi-Channel Attention Layer
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, edges_in_d=0,
                 act_fn=nn.SiLU(), dropout=0.1, time_emb_dim = None,**kwargs):
        super().__init__()
        self.hidden_nf = hidden_nf

        self.dropout = nn.Dropout(dropout)

        self.linear_q = nn.Linear(input_nf, hidden_nf)
        self.linear_kv = nn.Linear(input_nf + 2*n_channel ** 2 + edges_in_d, hidden_nf * 2)  # parallel calculate kv

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_nf * 2)
        ) if exists(time_emb_dim) else None
        
        # rot_layer = nn.Linear(hidden_nf, n_channel, bias=False)
        # torch.nn.init.xavier_uniform_(rot_layer.weight, gain=0.001)
        # rot_mlp = []
        # rot_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        # rot_mlp.append(act_fn)
        # rot_mlp.append(rot_layer)
        # self.rot_mlp = nn.Sequential(*rot_mlp)
        
        # self.to_rot = nn.Linear(n_channel * 3, 3, bias=True)
        # with torch.no_grad():
        #     self.to_rot.weight.fill_(0.0)
            
    def att_model(self, h, edge_index, radial, edge_attr):
        '''
        :param h: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param radial: [n_edge, n_channel, n_channel]
        :param edge_attr: [n_edge, edge_dim]
        '''
        row, col = edge_index
        source, target = h[row], h[col]  # [n_edge, input_size]

        # qkv
        q = self.linear_q(source)  # [n_edge, hidden_size]
        n_channel = radial.shape[1]
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, n_channel ^ 2]
        if edge_attr is not None:
            target_feat = torch.cat([radial, target, edge_attr], dim=1)
        else:
            target_feat = torch.cat([radial, target], dim=1)
        kv = self.linear_kv(target_feat)  # [n_edge, hidden_size * 2]
        k, v = kv[..., 0::2], kv[..., 1::2]  # [n_edge, hidden_size]

        # attention weight
        alpha = torch.sum(q * k, dim=1)  # [n_edge]

        # print(f'alpha1, {torch.isnan(alpha).sum()}', )

        # alpha = scatter_softmax(alpha, row, h.shape[0]) # [n_edge]
        alpha = scatter_softmax(alpha, row) # [n_edge]

        # print(f'alpha2, {torch.isnan(alpha).sum()}', )

        return alpha, v
        
    def node_model(self, h, edge_index, att_weight, v):
        '''
        :param h: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param att_weight: [n_edge, 1], unsqueezed before passed in
        :param v: [n_edge, hidden_size]
        '''
        row, _ = edge_index
        agg = unsorted_segment_sum(att_weight * v, row, h.shape[0])  # [bs * n_node, hidden_size]
        agg = self.dropout(agg)
        return h + agg  # residual

    def coord_model(self, coord, edge_index, coord_diff, att_weight, v):
        '''
        :param coord: [bs * n_node, n_channel, d]
        :param edge_index: list of [n_edge], [n_edge]
        :param coord_diff: [n_edge, n_channel, d]
        :param att_weight: [n_edge, 1], unsqueezed before passed in
        :param v: [n_edge, hidden_size]
        '''
        row, _ = edge_index
        coord_v = att_weight * self.coord_mlp(v)  # [n_edge, n_channel]
        trans = coord_diff * coord_v.unsqueeze(-1)
        agg = unsorted_segment_sum(trans, row, coord.size(0))
        coord = coord + agg
        return coord

    # def rot_model(self, rot, edge_index, coord_diff, att_weight, v):
    #     '''
    #     :param coord: [bs * n_node, n_channel, d]
    #     :param edge_index: list of [n_edge], [n_edge]
    #     :param coord_diff: [n_edge, n_channel, d]
    #     :param att_weight: [n_edge, 1], unsqueezed before passed in
    #     :param v: [n_edge, hidden_size]
    #     '''
    #     row, _ = edge_index
    #     coord_v = att_weight * self.rot_mlp(v)  # [n_edge, n_channel]
    #     trans = coord_diff * coord_v.unsqueeze(-1) # [n_edge, n_channel, d]
    #     agg = unsorted_segment_sum(trans, row, rot.size(0)) # [bs * n_node, n_channel, d]
    #     n_nodes = rot.shape[0]
    #     agg = agg.reshape(n_nodes, -1) # [bs * n_node, n_channel * d]
    #     rot_update_vec = self.to_rot(agg) # [bs * n_node, 3]
    #     quat = rot_to_quat(rot) # [bs * n_node, 4]
    #     rot = rot @ quaternion_1ijk_to_rotation_matrix(rot_update_vec)
    #     # rot = compose_q_update_vec(quat, rot_update_vec) # [bs * n_node, 3, 3]
    #     return rot
    
    def forward(self, h, edge_index, coord, edge_attr=None,time_emb=None):
        radial, dist, coord_diff = coord2radial(edge_index, coord)
        radial = torch.cat([radial, dist], dim=-1) # [n_edge, n_channel, n_channel * 2]
        att_weight, v = self.att_model(h, edge_index, radial, edge_attr)

        # print(f'att_weight, {torch.isnan(att_weight).sum()}', )
        # print(f'v, {torch.isnan(v).sum()}', )

        flat_att_weight = att_weight
        att_weight = att_weight.unsqueeze(-1)  # [n_edge, 1]
        h = self.node_model(h, edge_index, att_weight, v)
        coord = self.coord_model(coord, edge_index, coord_diff, att_weight, v)
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            scale, shift = time_emb.chunk(2, dim = 1)
            h = h * (scale + 1) + shift
        return h, coord, flat_att_weight


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class SeqGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, in_edge_nf=0,
                 act_fn=nn.SiLU(), residual=True, dropout=0.1,
                 ):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.dropout = nn.Dropout(dropout)
        


        self.linear_q = nn.Linear(in_node_nf, hidden_nf)
        self.linear_kv = nn.Linear(in_node_nf + 2*n_channel ** 2 + in_edge_nf, hidden_nf * 2)  # parallel calculate kv
        embed_dim = int(0.5*hidden_nf)
        self.hydropathy_embed = nn.Embedding(10, embed_dim, padding_idx=0)    # 1: hydrophilic, 2: moderate, 3. unknown
        self.charge_embed = nn.Embedding(10, embed_dim, padding_idx=0)    # 1: positive, 2: negative, 3. unknown
        fea_dim = embed_dim + embed_dim + hidden_nf # hydropathy + charge + h
        
        self.mlp = nn.Sequential(
            nn.Linear(fea_dim, fea_dim), 
            nn.ReLU(),
            nn.Linear(fea_dim, fea_dim),
            nn.ReLU(),
            nn.Linear(fea_dim, out_node_nf)
        )
        
        

    def att_model(self, h, edge_index, radial, edge_attr):
        '''
        :param h: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param radial: [n_edge, n_channel, n_channel]
        :param edge_attr: [n_edge, edge_dim]
        '''
        row, col = edge_index
        source, target = h[row], h[col]  # [n_edge, input_size]

        # qkv
        q = self.linear_q(source)  # [n_edge, hidden_size]
        n_channel = radial.shape[1]
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, n_channel ^ 2]
        if edge_attr is not None:
            target_feat = torch.cat([radial, target, edge_attr], dim=1)
        else:
            target_feat = torch.cat([radial, target], dim=1)
        kv = self.linear_kv(target_feat)  # [n_edge, hidden_size * 2]
        k, v = kv[..., 0::2], kv[..., 1::2]  # [n_edge, hidden_size]

        # attention weight
        alpha = torch.sum(q * k, dim=1)  # [n_edge]

        # print(f'alpha1, {torch.isnan(alpha).sum()}', )

        # alpha = scatter_softmax(alpha, row, h.shape[0]) # [n_edge]
        alpha = scatter_softmax(alpha, row) # [n_edge]

        # print(f'alpha2, {torch.isnan(alpha).sum()}', )

        return alpha, v
        
    def node_model(self, h, edge_index, att_weight, v):
        '''
        :param h: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param att_weight: [n_edge, 1], unsqueezed before passed in
        :param v: [n_edge, hidden_size]
        '''
        row, _ = edge_index
        agg = unsorted_segment_sum(att_weight * v, row, h.shape[0])  # [bs * n_node, hidden_size]
        agg = self.dropout(agg)
        return h + agg  # residual
    
    def forward(self, h, edge_index, coord, edge_attr, hydropathy, charge):
        radial, dist, coord_diff = coord2radial(edge_index, coord)
        radial = torch.cat([radial, dist], dim=-1)
        att_weight, v = self.att_model(h, edge_index, radial, edge_attr)
        att_weight = att_weight.unsqueeze(-1)  # [n_edge, 1]
        h = self.node_model(h, edge_index, att_weight, v)
        
        hydropathy_feat = self.hydropathy_embed(hydropathy)
        charge_feat = self.charge_embed(charge)

        h = torch.cat([h, hydropathy_feat, charge_feat], dim=-1)
        h = self.mlp(h)
        return h


class MCAttEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1, dense=False,self_condition=False):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.self_condition = self_condition
        self.dropout = nn.Dropout(dropout)
        in_c = in_node_nf * 2 if self_condition else in_node_nf
        self.linear_in = nn.Linear(in_c, self.hidden_nf)

        
        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', MC_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout,time_emb_dim=self.hidden_nf,
            ))
            
            # TODO: add parameter for passing edge type to interaction layer
            self.add_module(f'att_{i}', MC_Att_L(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                n_channel, edges_in_d=0, act_fn=act_fn, dropout=dropout,time_emb_dim=self.hidden_nf,
            ))
        self.out_layer = MC_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
            edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,time_emb_dim=self.hidden_nf,
        )
        self.time_embedding = nn.Linear(self.hidden_nf + 3, self.hidden_nf)
        # embed_dim = int(0.5*self.hidden_nf)
        # self.hydropathy_embed = nn.Embedding(10, embed_dim, padding_idx=0)    # 1: hydrophilic, 2: moderate, 3. unknown
        # self.charge_embed = nn.Embedding(10, embed_dim, padding_idx=0)    # 1: positive, 2: negative, 3. unknown
        # fea_dim =  embed_dim + embed_dim + hidden_nf # hydropathy + charge + h
        # self.linear_embed = nn.Linear(fea_dim, hidden_nf)
        
        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        # self.rot_net = nn.Sequential(
        #     nn.Linear(self.hidden_nf, self.hidden_nf), nn.ReLU(),
        #     nn.Linear(self.hidden_nf, self.hidden_nf), nn.ReLU(),
        #     nn.Linear(self.hidden_nf, 3)
        # )
        # sinu_pos_emb = SinusoidalPosEmb(self.hidden_nf)

        # self.time_mlp = nn.Sequential(
        #     sinu_pos_emb,
        #     nn.Linear(self.hidden_nf, self.hidden_nf),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_nf, self.hidden_nf)
        # )
        
        
    def forward(
        self, h, x, ctx_edges, att_edges, condition,  
        class_condition, hydropathy, charge,
        h_self_cond=None, ctx_edge_attr=None, att_edge_attr=None, 
        return_attention=False, return_h=False,
        ):
        if self.self_condition:
            h_self_cond = torch.zeros_like(h) if h_self_cond is None else h_self_cond
            h = torch.cat((h_self_cond, h), dim = -1)
        h = self.linear_in(h)
        h = self.dropout(h)
        
        # t = self.time_mlp(condition)
        
        ctx_states, ctx_coords, atts = [], [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](h, ctx_edges, x, edge_attr=ctx_edge_attr,time_emb=None)
            ctx_states.append(h)
            ctx_coords.append(x)
            h, x, att = self._modules[f'att_{i}'](h, att_edges, x, edge_attr=att_edge_attr,time_emb=None)
            atts.append(att)
            
        t_embed = torch.stack([condition, torch.sin(condition), torch.cos(condition)], dim=-1)
        h = self.time_embedding(torch.cat([h, t_embed], dim=-1))
        
        # # predict rotation
        # rot_vec_update = self.rot_net(h)
        # rot_mat_update = quaternion_1ijk_to_rotation_matrix(rot_vec_update)
        # rot = rot @ rot_mat_update
        
        # predict ca position
        h, x = self.out_layer(h, ctx_edges, x, edge_attr=ctx_edge_attr,time_emb=None)
        hidden = h
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            
        # predict aa
        # hydropathy_feat = self.hydropathy_embed(hydropathy)
        # charge_feat = self.charge_embed(charge)
        # h = torch.cat([h, hydropathy_feat, charge_feat], dim=-1)
        # h = self.linear_embed(h)
        h = self.dropout(h)
        h = self.linear_out(h) # [*, num_aa]
        out = (h, x)
        if return_attention:
            out = out + (atts,)
        if return_h:
            out = out + (hidden,)
        return out


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    dist = torch.cdist(coord[row], coord[col])  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    dist = F.normalize(dist, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, dist, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 10
    x_dim = 3
    n_channel = 5

    # Dummy variables h, x and fully connected edges
    h = torch.randn(batch_size *  n_nodes, n_feat)
    x = torch.randn(batch_size * n_nodes, n_channel, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    ctx_edges, att_edges = edges, edges

    # Initialize EGNN
    gnn = MCAttEGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=21, n_channel=n_channel)

    # Run EGNN
    h, x = gnn(h, x, ctx_edges, att_edges)

    print(h)
    print(x)