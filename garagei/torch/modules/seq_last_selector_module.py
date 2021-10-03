import copy
import numpy as np
import torch
from torch import nn

import dowel_wrapper
from iod.utils import zip_dict

class SeqLastSelectorModule(nn.Module):

    def __init__(self,
                 *,
                 inner_module,
                 input_dim=None,
                 use_delta=False,
                 omit_obs_idxs=None,
                 restrict_obs_idxs=None,
                 batch_norm=None,
                 noise=None,
                 ):
        super().__init__()

        if restrict_obs_idxs is not None:
            input_dim = len(restrict_obs_idxs)

        self.inner_module = inner_module
        self.input_dim = input_dim
        self.use_delta = use_delta
        self.omit_obs_idxs = omit_obs_idxs
        self.restrict_obs_idxs = restrict_obs_idxs
        self.batch_norm = batch_norm
        self.noise = noise

        if self.batch_norm:
            self.input_batch_norm = torch.nn.BatchNorm1d(self.input_dim, momentum=0.01, affine=False)
            self.input_batch_norm.eval()

    def _process_input(self, x):
        # x: (Batch (list), Time, Dim)

        assert isinstance(x, list)
        batch_size = len(x)

        x = [e[-1:] for e in x]

        if self.omit_obs_idxs is not None:
            for i in range(batch_size):
                x[i] = x[i].clone()
                x[i][:, self.omit_obs_idxs] = 0

        if self.restrict_obs_idxs is not None:
            for i in range(batch_size):
                x[i] = x[i][:, self.restrict_obs_idxs]

        if self.use_delta:
            for i in range(batch_size):
                x[i] = x[i][1:] - x[i][:-1]
                if self.output_type in ['BatchTime_Hidden', 'BatchTimecummean_Hidden']:
                    x[i] = torch.cat([x[i][0:1], x[i]])

        if self.batch_norm:
            x_cat = torch.cat(x, dim=0)
            x_cat = self.input_batch_norm(x_cat)
            x = list(x_cat.split([len(x_i) for x_i in x], dim=0))

        if self.noise is not None and self.training:
            for i in range(batch_size):
                x[i] = x[i] + torch.randn_like(x[i], device=x[0].device) * self.noise

        x = torch.cat(x, dim=0)
        return x

    def forward(self, x, **kwargs):
        return self.inner_module(self._process_input(x), **kwargs), (None, None)

    def forward_with_transform(self, x, *, transform, **kwargs):
        return self.inner_module.forward_with_transform(self._process_input(x),
                                                        transform=transform,
                                                        **kwargs), (None, None)

    def forward_with_chunks(self, x, *, merge, **kwargs):
        outs = []
        for chunk_x, chunk_kwargs in zip(x, zip_dict(kwargs)):
            chunk_out = self.inner_module(chunk_x, **chunk_kwargs)
            outs.append(chunk_out)
        return self.inner_module.forward_with_chunks(outs, merge=merge), (None, None)

    def forward_force_only_last(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def get_last_linear_layers(self):
        return self.inner_module.get_last_linear_layers()

