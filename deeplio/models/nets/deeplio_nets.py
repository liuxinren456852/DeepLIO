
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepIO(nn.Module):
    def __init__(self, cfg):
        super(DeepIO, self).__init__()

        self.p = cfg.get('dropout', 0.)
        feat_net_cfg = cfg['feature-net']
        self.feat_net_type = feat_net_cfg.get('name', 'deepio0').lower()

        if self.feat_net_type == 'deepio0':
            self.feat_net = DeepIOFeat0(feat_net_cfg)
        elif self.feat_net_type == 'deepio1':
            self.feat_net = DeepIOFeat1(feat_net_cfg)

        if self.p > 0.:
            self.drop_out = nn.Dropout(self.p)

        # get featurenet output shape
        data = torch.rand((1, 2, 2, self.feat_net.input_size))
        self.feat_net.eval()
        with torch.no_grad():
            out = self.feat_net(data)
        n_feat_out = out.shape[-1]

        self.fc1 = nn.Linear(n_feat_out, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_pos = nn.Linear(256, 3)
        self.fc_ori = nn.Linear(256, 4)

    def forward(self, x):
        x = self.feat_net(x)

        x = F.relu(self.fc1(x), inplace=True)
        x = self.bn1(x)
        if self.p > 0.:
            x = self.drop_out(x)

        x_pos = self.fc_pos(x)
        x_ori = self.fc_ori(x)
        return x_pos, x_ori

    @property
    def name(self):
        res = "{}_{}".format(self.__class__.__name__, self.feat_net.__class__.__name__)
        return res


class DeepIOFeat0(nn.Module):
    def __init__(self, cfg):
        super(DeepIOFeat0, self).__init__()

        self.input_size = cfg['input-size']
        hidden_size = cfg.get('hidden-size', [6, 6])
        num_layers = cfg.get('num-layers', 2)

        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size[0]))
        for i in range(1, num_layers):
            l = nn.Linear(hidden_size[i-1], hidden_size[i])
            layers.append(l)
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        n_batches = len(x)
        n_seq = len(x[0]) # all seq. are the same length

        outputs = []
        for b in range(n_batches):
            for s in range(n_seq):
                y = x[b][s]
                for m in self.net:
                    y = F.relu(m(y))
                outputs.append(torch.sum(y, dim=0))
        outputs = torch.stack(outputs)
        return outputs


class DeepIOFeat1(nn.Module):
    def __init__(self, cfg):
        super(DeepIOFeat1, self).__init__()
        self.input_size = cfg['input-size']
        self.hidden_size = cfg.get('hidden-size', [6, 6])
        num_layers = cfg.get('num-layers', 2)
        self.bidirectional = cfg.get('bidirectional', False)

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0],
                           num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional)

        self.num_dir = 2 if self.bidirectional else 1

    def forward(self, x):
        x_all = [xx for x_ in x for xx in x_]
        lengths = [x_.size(0) for x_ in x_all]
        x_padded = nn.utils.rnn.pad_sequence(x_all, batch_first=True)
        b, s, n = x_padded.shape

        x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(x_padded)
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.view(b, s, self.num_dir, self.hidden_size[0])
        out = out[:, -1, 0]
        return out

