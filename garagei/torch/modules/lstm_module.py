import copy
import numpy as np
import torch
from torch import nn

import dowel_wrapper
from iod.utils import zip_dict

class LSTMModule(nn.Module):

    def __init__(self,
                 *,
                 input_dim,
                 hidden_dim,
                 num_layers,
                 output_type,
                 pre_lstm_module,
                 post_lstm_module,  # e.g. GaussianMLPModule, GaussianMLPIndependentStdModule, GaussianMLPTwoHeadedModule
                 bidirectional=True,
                 num_reduced_obs=None,
                 use_delta=False,
                 omit_obs_idxs=None,
                 restrict_obs_idxs=None,
                 batch_norm=None,
                 noise=None,
                 disable_lstm=False,
                 ):
        super().__init__()

        if restrict_obs_idxs is not None:
            input_dim = len(restrict_obs_idxs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_type = output_type
        self.bidirectional = bidirectional
        self.pre_lstm_module = pre_lstm_module
        self.post_lstm_module = post_lstm_module
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        self.num_reduced_obs = num_reduced_obs
        self.use_delta = use_delta
        self.omit_obs_idxs = omit_obs_idxs
        self.restrict_obs_idxs = restrict_obs_idxs
        self.batch_norm = batch_norm
        self.noise = noise
        self.disable_lstm = disable_lstm

        if self.batch_norm:
            self.input_batch_norm = torch.nn.BatchNorm1d(self.input_dim, momentum=0.01, affine=False)
            self.input_batch_norm.eval()

    def _forward_lstm(self,
                      x,
                      hidden_cell_state_tuple=None,
                      grad_window_size=None,
                      force_only_last=False,
                      ):
        # x: (Batch (list), Time, Dim)
        # hidden_cell_state_tuple: (Layer * Direction, Batch, Hidden), (Layer * Direction, Batch, Hidden)

        assert isinstance(x, list)
        batch_size = len(x)

        x = copy.copy(x)

        if self.omit_obs_idxs is not None:
            for i in range(batch_size):
                x[i] = x[i].clone()
                x[i][:, self.omit_obs_idxs] = 0

        if self.restrict_obs_idxs is not None:
            for i in range(batch_size):
                x[i] = x[i][:, self.restrict_obs_idxs]

        if self.num_reduced_obs is not None:
            for i in range(batch_size):
                if x[i].size(0) <= self.num_reduced_obs:
                    continue
                idxs = np.flip(np.round(
                    np.linspace(x[i].size(0) - 1, 0, self.num_reduced_obs)
                ).astype(int)).copy()
                x[i] = x[i][idxs]

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

        if grad_window_size is not None:
            # Currently only support when all of the sequence lengths are same (to prevent empty sequence packing)
            if min([x_i.size(0) for x_i in x]) != max([x_i.size(0) for x_i in x]):
                raise NotImplementedError()

        if self.disable_lstm:
            # Identity layer
            stacked_x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
            stacked_x, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(stacked_x, padding_value=0.0)
            lstm_out = torch.zeros(stacked_x.size(0), stacked_x.size(1), self.hidden_dim, device=x[0].device)
            assert stacked_x.size(2) <= lstm_out.size(2)
            lstm_out[:, :, :stacked_x.size(2)] = stacked_x

            # lstm_out: (Time, Batch, Hidden)
            hidden = torch.zeros(self.num_layers, lstm_out.size(1), lstm_out.size(2), device=x[0].device)
            cell_state = torch.zeros(self.num_layers, lstm_out.size(1), lstm_out.size(2), device=x[0].device)
        elif grad_window_size is None or grad_window_size == min([x_i.size(0) for x_i in x]):
            # Allow gradient flow in all of the indices

            x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
            assert isinstance(x, torch.nn.utils.rnn.PackedSequence)

            lstm_out, (hidden, cell_state) = self.lstm(x, hidden_cell_state_tuple)

            assert isinstance(lstm_out, torch.nn.utils.rnn.PackedSequence)
            lstm_out, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, padding_value=0.0)
        else:
            # Freeze [:-grad_window_size] so that allow gradient flow only in [-grad_window_size:]

            assert min([x_i.size(0) for x_i in x]) >= grad_window_size
            if self.bidirectional:
                raise NotImplementedError()

            x_left = [x_i[:-grad_window_size] for x_i in x]
            x_right = [x_i[-grad_window_size:] for x_i in x]

            with torch.no_grad():
                x_left = torch.nn.utils.rnn.pack_sequence(x_left, enforce_sorted=False)
                lstm_out_left, (hidden, cell_state) = self.lstm(x_left, hidden_cell_state_tuple)
                lstm_out_left, seq_lengths_left = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_left, padding_value=0.0)

            x_right = torch.nn.utils.rnn.pack_sequence(x_right, enforce_sorted=False)
            lstm_out_right, (hidden, cell_state) = self.lstm(x_right, (hidden, cell_state))
            lstm_out_right, seq_lengths_right = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_right, padding_value=0.0)

            lstm_out = torch.cat([lstm_out_left, lstm_out_right], dim=0)
            seq_lengths = seq_lengths_left + seq_lengths_right

        if self.output_type == 'Batch_Hidden' or (self.output_type == 'BatchTimecummean_Hidden' and force_only_last):
            lstm_out_mean = lstm_out.sum(dim=0) / seq_lengths.to(torch.float32).to(lstm_out.device)[:, None]
            assert lstm_out_mean.size(0) == batch_size
            return lstm_out_mean, (hidden, cell_state)
        elif self.output_type == 'Batchlast_Hidden' or (self.output_type == 'BatchTime_Hidden' and force_only_last):
            if seq_lengths.min() != seq_lengths.max():
                raise NotImplementedError()
            lstm_out_last = lstm_out[-1, :, :]
            assert lstm_out_last.size(0) == batch_size
            return lstm_out_last, (hidden, cell_state)
        elif self.output_type in ['BatchTime_Hidden', 'BatchTimecummean_Hidden']:
            if self.output_type in ['BatchTimecummean_Hidden']:
                if self.bidirectional:
                    raise NotImplementedError()
                lstm_out = (lstm_out.cumsum(dim=0)
                            / torch.arange(1.0, 1.0 + seq_lengths.max(),
                                           device=lstm_out.device)[:, None, None])

            # lstm_out: (Time, Batch, Hidden)
            lstm_out = lstm_out.transpose(0, 1)
            # lstm_out: (Batch, Time, Hidden)
            lstm_out = lstm_out.reshape(lstm_out.size(0) * lstm_out.size(1), *lstm_out.size()[2:])
            # lstm_out: (Batch * Time, Hidden)
            return lstm_out, (hidden, cell_state)
        else:
            assert False

    def _forward_pre_lstm_module(self, x):
        if self.pre_lstm_module is None:
            return x

        orig_lengths = None
        if isinstance(x, list):
            orig_lengths = [len(i) for i in x]
            x = torch.cat(x, dim=0)
        res = self.pre_lstm_module(x)
        if orig_lengths is not None:
            res = list(torch.split(res, orig_lengths, dim=0))
        return res

    def forward(self, x, **kwargs):
        x = self._forward_pre_lstm_module(x)
        if self.disable_lstm and not isinstance(x, list):
            # Manually create lstm_out
            if self.omit_obs_idxs is not None:
                x = x.clone()
                x[:, self.omit_obs_idxs] = 0
            lstm_out = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            assert x.size(1) <= lstm_out.size(1)
            lstm_out[:, :x.size(1)] = x
            hidden_cell_state_tuple = None
        else:
            lstm_out, hidden_cell_state_tuple = self._forward_lstm(x, **kwargs)
        out = self.post_lstm_module(lstm_out)
        return out, hidden_cell_state_tuple

    def forward_with_transform(self, x, *, transform, **kwargs):
        x = self._forward_pre_lstm_module(x)
        lstm_out, hidden_cell_state_tuple = self._forward_lstm(x, **kwargs)
        out = self.post_lstm_module.forward_with_transform(lstm_out, transform=transform)
        return out, hidden_cell_state_tuple

    def forward_with_chunks(self, x, *, merge, **kwargs):
        x = self._forward_pre_lstm_module(x)
        lstm_out = []
        hidden = []
        cell_state = []
        for chunk_x, chunk_kwargs in zip(x, zip_dict(kwargs)):
            chunk_lstm_out, (chunk_hidden, chunk_cell_state) = self._forward_lstm(chunk_x, **chunk_kwargs)
            lstm_out.append(chunk_lstm_out)
            hidden.append(chunk_hidden)
            cell_state.append(chunk_cell_state)
        out = self.post_lstm_module.forward_with_chunks(lstm_out, merge=merge)
        hidden = merge(hidden, batch_dim=1)
        cell_state = merge(cell_state, batch_dim=1)
        return out, (hidden, cell_state)

    def forward_force_only_last(self, x, **kwargs):
        x = self._forward_pre_lstm_module(x)
        lstm_out, hidden_cell_state_tuple = self._forward_lstm(x, force_only_last=True, **kwargs)
        out = self.post_lstm_module(lstm_out)
        return out, hidden_cell_state_tuple

    def get_last_linear_layers(self):
        return self.post_lstm_module.get_last_linear_layers()

