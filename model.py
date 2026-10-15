import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import OptTensor
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing


class GraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

    def gcn_norm(self, edge_index, num_nodes):
        edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        edge_index = edge_index.long()
        edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j


class Fusion(nn.Module):
    def __init__(self, emb_dim, bsg_types, trbg_types):
        super(Fusion, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types
        self.trbg_types = trbg_types
        self.proj_layers = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim, bias=True),
            nn.Linear(emb_dim, emb_dim, bias=True)
        ])

    def reset_parameters(self):
        for layer in self.proj_layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, emb_dict, key):
        for i, behavior_types in enumerate([self.bsg_types, self.trbg_types]):
            proj_key = self.proj_layers[i](key)
            query_embs = torch.stack([emb_dict[bt] for bt in behavior_types], dim=1)
            scores = torch.einsum('nd,nbd->nb', proj_key, query_embs) / (self.emb_dim ** 0.5)
            attention = scores.softmax(dim=1).unsqueeze(-1)
            key = (attention * query_embs).sum(dim=1)

        return key


class Denoiser(nn.Module):
    def __init__(self, emb_dim):
        super(Denoiser, self).__init__()
        self.emb_dim = emb_dim
        self.in_layer = nn.Linear(4 * emb_dim, emb_dim, bias=True)
        self.out_layer = nn.Linear(emb_dim, emb_dim, bias=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_layer.weight)
        nn.init.zeros_(self.in_layer.bias)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def timestep_embedding(self, t):
        half = self.emb_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_t, t, h, z):
        tau = self.timestep_embedding(t + 1)
        inp = torch.cat([x_t, tau, h, z], dim=-1)
        return self.out_layer(F.silu(self.in_layer(inp)))


class CausalDiffusion(nn.Module):
    def __init__(self, emb_dim, steps, noise_scale, noise_min, noise_max, sampling_steps):
        super(CausalDiffusion, self).__init__()
        self.steps = steps
        self.sampling_steps = sampling_steps
        self.denoiser = Denoiser(emb_dim)

        betas = torch.linspace(noise_scale * noise_min, noise_scale * noise_max, steps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('posterior_coef1',
                             (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float())
        self.register_buffer('posterior_coef2',
                             ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float())

    def reset_parameters(self):
        self.denoiser.reset_parameters()

    def q_sample(self, x_0, t, noise):
        a = self.alphas_cumprod[t].unsqueeze(-1)
        return a.sqrt() * x_0 + (1.0 - a).sqrt() * noise

    def posterior_mean(self, x_0_hat, x_t, t):
        c1 = self.posterior_coef1[t].unsqueeze(-1)
        c2 = self.posterior_coef2[t].unsqueeze(-1)
        return c1 * x_0_hat + c2 * x_t

    def train_step(self, x_0, h, z):
        t = torch.randint(0, self.steps, (x_0.size(0),), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        x_0_hat = self.denoiser(x_t, t, h, z)
        return x_0_hat, t

    def sample(self, x_0, h, z):
        if self.sampling_steps == 0:
            t = torch.zeros(x_0.size(0), dtype=torch.long, device=x_0.device)
            return self.denoiser(x_0, t, h, z)
        t = torch.full((x_0.size(0),), self.sampling_steps - 1, dtype=torch.long, device=x_0.device)
        x_t = self.q_sample(x_0, t, torch.zeros_like(x_0))
        for step in range(self.sampling_steps - 1, -1, -1):
            t = torch.full((x_0.size(0),), step, dtype=torch.long, device=x_0.device)
            x_0_hat = self.denoiser(x_t, t, h, z)
            x_t = self.posterior_mean(x_0_hat, x_t, t)
        return x_t

    def loss(self, x_0, x_0_hat, t, index):
        x_0, x_0_hat, t = x_0[index], x_0_hat[index], t[index]

        snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        snr_prev = self.alphas_cumprod_prev / (1.0 - self.alphas_cumprod_prev).clamp(min=1e-8)
        weight = snr_prev[t] - snr[t]
        weight = torch.where(t == 0, torch.ones_like(weight), weight)
        rec = (weight * (x_0 - x_0_hat).pow(2).mean(dim=-1)).mean()
        return rec


class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class Model(nn.Module):
    def __init__(self, data, args):
        super(Model, self).__init__()
        self.edge_dict = data['edge_dict']

        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = args.emb_dim
        self.temperature = 1
        self.dropout = nn.Dropout(0.1)

        self.bsg_types = data['bsg_types']
        self.tcb_types = data['tcb_types']
        self.tib_types = data['tib_types']
        self.trbg_types = self.tcb_types + self.tib_types
        self.total_behaviors = ['base'] + self.bsg_types + self.trbg_types
        self.target_behavior = 'buy'
        self.aux_types = [bt for bt in self.bsg_types if bt != self.target_behavior]
        n_nodes = self.n_users + self.n_items + 2
        deg = torch.stack([torch.bincount(self.edge_dict[bt][0].long(), minlength=n_nodes).float()
                           for bt in self.aux_types], dim=1)
        propensity = deg / deg.sum(dim=1, keepdim=True).clamp(min=1)
        active = (deg.sum(dim=1) > 0).float()
        self.register_buffer('propensity', propensity)
        self.register_buffer('active', active.unsqueeze(-1))
        self.register_buffer('bias_prior', propensity.sum(dim=0) / active.sum().clamp(min=1))

        self.alpha = args.alpha
        self.gamma = args.gamma

        self.fusion = Fusion(self.emb_dim, self.bsg_types, self.trbg_types)
        self.ce_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

        self.user_embedding = nn.Embedding(self.n_users+1, self.emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items+1, self.emb_dim, padding_idx=0)

        self.diffusion = CausalDiffusion(self.emb_dim, args.diff_steps, args.noise_scale, args.noise_min,
                                         args.noise_max, args.sampling_steps)

        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            if behavior_type in self.tcb_types:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(self.emb_dim, self.emb_dim) for _ in range(3)])
            else:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(self.emb_dim, self.emb_dim) for _ in range(1)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.fusion.reset_parameters()
        self.diffusion.reset_parameters()

    def propagate(self, x, edge_index, behavior_type):
        result = [x]
        for conv in self.convs[behavior_type]:
            x = conv(x, edge_index)
            x = F.normalize(x, dim=1)
            result.append(x)
        result = torch.stack(result, dim=1)
        x = result.sum(dim=1)
        return x

    def forward(self):
        edge_dict = self.edge_dict
        emb_dict = dict()

        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        base_emb = self.propagate(init_emb, edge_dict['base'], 'base')
        emb_dict['base'] = base_emb

        for behavior_type in self.bsg_types:
            emb_dict[behavior_type] = self.propagate(
                self.dropout(base_emb), edge_dict[behavior_type], behavior_type)

        target_emb = emb_dict[self.target_behavior]

        bsg_embs = torch.stack([emb_dict[bt] for bt in self.bsg_types], dim=1)
        scores = torch.einsum('nd,nbd->nb', target_emb, bsg_embs) / (self.emb_dim ** 0.5)
        attention = scores.softmax(dim=1).unsqueeze(-1)
        condition = base_emb + (attention * bsg_embs).sum(dim=1)

        behavior_embs = torch.stack([
            (self.propensity[:, k:k+1] * emb_dict[bt]).sum(dim=0) / self.propensity[:, k].sum().clamp(min=1e-12)
            for k, bt in enumerate(self.aux_types)], dim=0)
        bias_mean = (self.bias_prior @ behavior_embs).unsqueeze(0)

        x_0 = target_emb
        if self.training:
            bias_ctx = self.active * (self.propensity @ behavior_embs) + (1 - self.active) * bias_mean
            x_0_hat, t = self.diffusion.train_step(x_0, condition, bias_ctx)
            emb_dict['diffusion'] = (x_0, x_0_hat, t)
            target_emb = x_0_hat
        else:
            target_emb = self.diffusion.sample(x_0, condition, bias_mean.expand_as(x_0))

        for behavior_type in self.trbg_types:
            if self.target_behavior in behavior_type:
                previous_behavior = behavior_type.split('_')[0]
                emb_dict[behavior_type] = self.propagate(
                    self.dropout(emb_dict[previous_behavior]), edge_dict[behavior_type], behavior_type)

        emb_dict['final'] = self.fusion(emb_dict, target_emb)

        return emb_dict

    def loss(self, users, pos_idx, neg_idx):
        emb_dict = self.forward()
        user_emb, item_emb = torch.split(emb_dict['final'], [self.n_users+1, self.n_items+1], dim=0)

        cl_loss = torch.tensor(0.0).cuda()
        for behavior in self.bsg_types + ['base']:
            user, item = torch.split(emb_dict[behavior], [self.n_users+1, self.n_items+1], dim=0)
            cl_loss += self.alpha * self.cl_loss(user_emb[users], user[users])
            cl_loss += self.alpha * self.cl_loss(item_emb[pos_idx], item[pos_idx])

        causal_logits = torch.matmul(user_emb[users], item_emb[1:].transpose(0, 1))
        ce_loss = self.ce_loss(causal_logits, pos_idx - 1)
        reg_loss = self.reg_loss(user_emb[1:], item_emb[1:], require_pow=True)

        total_loss = ce_loss + cl_loss + 0.1 * reg_loss

        x_0, x_0_hat, t = emb_dict['diffusion']
        index = torch.cat([users, pos_idx + self.n_users + 1], dim=0)
        total_loss = total_loss + self.gamma * self.diffusion.loss(x_0, x_0_hat, t, index)

        return total_loss

    def cl_loss(self, x1, x2):
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()

    def predict(self, users):
        final_embeddings = self.forward()['final']
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users + 1, self.n_items + 1])

        user_emb = final_user_emb[users.long()]
        scores = torch.matmul(user_emb, final_item_emb.transpose(0, 1))
        scores[:, 0] = -float('inf')
        return scores