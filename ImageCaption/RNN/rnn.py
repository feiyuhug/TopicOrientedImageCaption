import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import numpy as np

class RNN(nn.Module):

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, \
                 dropout=0, hard_thresh=0.0, norm_thresh=0.0):
        super(RNN, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.hard_thresh = hard_thresh
        self.norm_thresh = norm_thresh
        self.dropout_state = {}

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
            w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
            b_ih = Parameter(torch.Tensor(gate_size))
            b_hh = Parameter(torch.Tensor(gate_size))
            layer_params = (w_ih, w_hh, b_ih, b_hh)

            param_names = ['weight_ih_l{}'.format(layer), 'weight_hh_l{}'.format(layer)]
            if bias:
                param_names += ['bias_ih_l{}'.format(layer), 'bias_hh_l{}{}'.format(layer)]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)
        self.reset_parameters__()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def orthogonal_initializer(self, weight_data, scale = 1.0):
        shape = weight_data.size()
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return torch.from_numpy(scale * q[:shape[0], :shape[1]]).type(weight_data.type())
    
    def bias_init(self, weight_data):
        re = torch.Tensor(*(weight_data.size()))
        re[:] = 0.01
        return re.type(weight_data.type())

    def reset_parameters_(self):
        for weights in self.all_weights:
            for weight in weights[0:2]:
                for i in range(4):
                    weight.data[self.hidden_size * i : self.hidden_size * (i+1)] = \
                            self.orthogonal_initializer(weight.data[self.hidden_size * i: \
                            self.hidden_size * (i+1)].transpose(1,0)).transpose(1,0)
            for bias in weights[2:4]:
                bias.data = self.bias_init(bias.data)
    
    def reset_parameters__(self):
        for weights in self.all_weights:
            for weight in weights[0:2]:
                weight.data.uniform_(-0.04, 0.04) 
            for bias in weights[2:4]:
                bias.data[:] = 0
    
    def group_lasso(self):
        #l1crit = nn.L1Loss(size_average=False)
        msecrit = nn.MSELoss(size_average=False)
        reg_loss = 0
        for param in self.all_weights:
            for weight in param[0:2]:
                reg_loss += msecrit(weight[0:3*self.hidden_size, :], target=torch.autograd.Variable(torch.zeros(weight[0:3*self.hidden_size, :].size()).type(weight.data.type()), requires_grad=False))
        return reg_loss
        '''
        reg = 0.0
        zero_counts = []
        for weights in self.all_weights:
            tmp_sum = 0.0
            for weight in weights[0:2]:
                tmp = torch.mul(weight, weight)
                tmp = tmp.sum(1)
                tmp1, tmp2, tmp3, tmp4 = tmp.chunk(4,0)
                tmp = tmp1 + tmp2 + tmp3 + tmp4
                tmp_sum += tmp
            zero_count = torch.sum(tmp_sum.data == 0)
            zero_counts.append(zero_count)
            tmp_sum = torch.sqrt(tmp_sum + 1e-8)
            reg += tmp_sum.sum()
        return reg, zero_counts
        '''

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def LSTMCell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, hard_thresh=0.0, norm_thresh=0.0, dropout_state=None):
        '''
        if input.is_cuda:
            igates = F.linear(input, w_ih)
            hgates = F.linear(hidden[0], w_hh)
            state = fusedBackend.LSTMFused.apply
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        '''
        hx, cx = hidden
        if dropout_state is not None:
            hdropout_state, idropout_state = dropout_state
            if idropout_state is not None:
                input = input * idropout_state.expand_as(input)
            if hdropout_state is not None:
                hx = hx * hdropout_state.expand_as(hx)
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
       
        # hard gate
        if norm_thresh > 0.0:
            #ingate_ = F.threshold(ingate, norm_thresh, 0)
            outgate_ = F.threshold(outgate, norm_thresh, 0)
        else:
            #ingate_ = ingate
            outgate_ = outgate
        
        #gate_norm = ingate_.sum() + outgate_.sum()
        #gate_norm_c = torch.sum(ingate_.data != 0) + torch.sum(outgate_.data != 0)
        gate_norm = outgate_.sum()
        gate_norm_c = torch.sum(outgate_.data != 0)

        if hard_thresh > 0.0:
            #ingate__ = F.threshold(ingate, hard_thresh, 0)
            outgate__ = F.threshold(outgate, hard_thresh, 0)
        else:
            #ingate__ = ingate
            outgate__ = outgate
        #hard_norm = ingate__.sum() + outgate__.sum()
        #hard_norm_c = torch.sum(ingate__.data != 0) + torch.sum(outgate__.data != 0)
        hard_norm = outgate__.sum()
        hard_norm_c = torch.sum(outgate__.data != 0)
        
        #ingate = ingate__
        outgate = outgate__
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy, (gate_norm, gate_norm_c), (hard_norm, hard_norm_c)

    def cell_forward(self, input, hidden):
        return self.LSTMCell(input, hidden, *(self.all_weights[0]), hard_thresh=self.hard_thresh, norm_thresh=self.norm_thresh)

    def Recurrent(self, inner, train=True):
        def forward(input, hidden, weight, batch_sizes, hard_thresh=0.0, norm_thresh=0.0):
            output = []
            steps = range(input.size(0))
            gate_norm_sum = None
            gate_norm_c_sum = None
            hard_norm_sum = None
            hard_norm_c_sum = None
            if train and self.dropout > 0.0:
                #hdropout_state = torch.autograd.Variable((torch.Tensor(self.hidden_size).bernoulli_(1 - self.dropout) / (1 - self.dropout)).type(input.data.type()), requires_grad=False)
                #idropout_state = torch.autograd.Variable((torch.Tensor(self.input_size).bernoulli_(1 - self.dropout) / (1 - self.dropout)).type(input.data.type()), requires_grad=False)
                #dropout_state = (hdropout_state, idropout_state)
                #dropout_state = (hdropout_state, None)
                dropout_state = None
            else:
                dropout_state = None
            for i in steps:
                hidden_h, hidden_c, gate_norm, hard_norm = inner(input[i], hidden, *weight, hard_thresh=hard_thresh, norm_thresh=norm_thresh, dropout_state=dropout_state)
                hidden = (hidden_h, hidden_c)
                output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
                gate_norm, gate_norm_c = gate_norm
                hard_norm, hard_norm_c = hard_norm
                if gate_norm_sum is not None:
                    gate_norm_sum += gate_norm
                    gate_norm_c_sum += gate_norm_c
                    hard_norm_sum += hard_norm
                    hard_norm_c_sum += hard_norm_c
                else:
                    gate_norm_sum = gate_norm
                    gate_norm_c_sum = gate_norm_c
                    hard_norm_sum = hard_norm
                    hard_norm_c_sum = hard_norm_c
            output = torch.cat(output, 0).view(input.size(0), *output[0].size())

            return hidden, output, (gate_norm_sum, gate_norm_c_sum), (hard_norm_sum, hard_norm_c_sum)

        return forward

    def VariableRecurrent(self, inner):
        def forward(input, hidden, weight, batch_sizes):

            output = []
            input_offset = 0
            last_batch_size = batch_sizes[0]
            hiddens = []
            flat_hidden = not isinstance(hidden, tuple)
            if flat_hidden:
                hidden = (hidden,)
            for batch_size in batch_sizes:
                step_input = input[input_offset:input_offset + batch_size]
                input_offset += batch_size

                dec = last_batch_size - batch_size
                if dec > 0:
                    hiddens.append(tuple(h[-dec:] for h in hidden))
                    hidden = tuple(h[:-dec] for h in hidden)
                last_batch_size = batch_size

                if flat_hidden:
                    hidden = (inner(step_input, hidden[0], *weight),)
                else:
                    hidden = inner(step_input, hidden, *weight)

                output.append(hidden[0])
            hiddens.append(hidden)
            hiddens.reverse()

            hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
            assert hidden[0].size(0) == batch_sizes[0]
            if flat_hidden:
                hidden = hidden[0]
            output = torch.cat(output, 0)

            return hidden, output

        return forward

    def variable_recurrent_factory(self, inner, train=True):
        return self.VariableRecurrent(inner)

    def StackedRNN(self, inners, num_layers, lstm=False, dropout=0, train=True, hard_thresh=0.0, norm_thresh=0.0):
        total_layers = num_layers

        def forward(input, hidden, weight, batch_sizes):
            assert(len(weight) == total_layers)
            next_hidden = []

            if lstm:
                hidden = list(zip(*hidden))
            
            gate_norm_sum = None
            gate_norm_c_sum = None
            hard_norm_sum = None
            hard_norm_c_sum = None

            for i in range(num_layers):
                all_output = []
                for j, inner in enumerate(inners):
                    l = i + j
                    hy, output, gate_norm, hard_norm = inner(input, hidden[l], weight[l], batch_sizes, hard_thresh, norm_thresh)
                    next_hidden.append(hy)
                    all_output.append(output)
                    gate_norm, gate_norm_c = gate_norm
                    hard_norm, hard_norm_c = hard_norm
                    if gate_norm_sum is not None:
                        gate_norm_sum += gate_norm
                        gate_norm_c_sum += gate_norm_c
                        hard_norm_sum += hard_norm
                        hard_norm_c_sum += hard_norm_c
                    else:
                        gate_norm_sum = gate_norm
                        gate_norm_c_sum = gate_norm_c
                        hard_norm_sum = hard_norm
                        hard_norm_c_sum = hard_norm_c

                input = torch.cat(all_output, input.dim() - 1)
                if dropout != 0 and i < num_layers - 1:
                    input = F.dropout(input, p=dropout, training=train, inplace=False)
                    '''
                    input_non_zero = torch.sum(input.data != 0)
                    input_all = torch.numel(input.data)
                    nonzero_ratio = float(input_non_zero) / float(input_all)
                    #dropout_ = max(1 - (1-dropout)/nonzero_ratio, 0)
                    dropout_exp = 1 - (1-dropout)/nonzero_ratio
                    dropout_ = np.random.normal(dropout_exp, 1.0)
                    #while dropout_ < 0.4 or dropout_ > 0.8:
                    while dropout_ < 0.5 or dropout_ > 0.8:
                        dropout_ = np.random.normal(dropout_exp, 1.0)
                    input = F.dropout(input, p=dropout_, training=train, inplace=False)
                    '''
            if lstm:
                next_h, next_c = zip(*next_hidden)
                next_hidden = (
                    torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                    torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
                )
            else:
                next_hidden = torch.cat(next_hidden, 0).view(
                    total_layers, *next_hidden[0].size()
                )

            return next_hidden, input, (gate_norm_sum, gate_norm_c_sum), (hard_norm_sum, hard_norm_c_sum)

        return forward

    def AutogradRNN(self, mode, input_size, hidden_size, num_layers=1, batch_first=False,
                    dropout=0, train=True, dropout_state=None, variable_length=False, hard_thresh=0.0, norm_thresh=0.0):
        if mode == 'LSTM':
            cell = self.LSTMCell
        else:
            raise Exception('Unknown mode: {}'.format(mode))
        rec_factory = self.variable_recurrent_factory if variable_length else self.Recurrent

        layer = (rec_factory(cell),)

        func = self.StackedRNN(layer, num_layers, (mode == 'LSTM'), dropout=dropout, train=train, hard_thresh=hard_thresh, norm_thresh=norm_thresh)

        def forward(input, weight, hidden, batch_sizes):
            if batch_first and not variable_length:
                input = input.transpose(0,1)

            nexth, output, gate_norm, hard_norm = func(input, hidden, weight, batch_sizes)

            if batch_first and not variable_length:
                output = output.transpose(0,1)

            return output, nexth, gate_norm, hard_norm

        return forward

    def rnn_backend(self, *args, **kwargs):

        def forward(input, *fargs, **fkwargs):
            func = self.AutogradRNN(*args, **kwargs)
            '''
            import torch
            if torch._C._jit_is_tracing(input):
                import torch.onnx.symbolic
                decorator = torch.onnx.symbolic_override_first_arg_based(
                    torch.onnx.symbolic.RNN_symbolic_builder(*args, **kwargs))
                func = decorator(func)
            '''
            return func(input, *fargs, **fkwargs)

        return forward

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_size[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = torch.autograd.Variable(input.data.new(self.num_layers,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(),
                                                        requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)
        func = self.rnn_backend(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            dropout_state=self.dropout_state,
            variable_length=is_packed,
            hard_thresh=self.hard_thresh,
            norm_thresh=self.norm_thresh
        )
        output, hidden, gate_norm, hard_norm = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)

        return output, hidden, gate_norm, hard_norm



