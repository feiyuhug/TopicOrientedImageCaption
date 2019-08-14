from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import numpy as np
from .CaptionModel import CaptionModel

class ShowTellModel_t(CaptionModel):
    def __init__(self, opt):
        super(ShowTellModel_t, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.topic_num = opt.topic_num

        self.ss_prob = 0.0 # Schedule sampling probability
        
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.topic_embed = nn.Embedding(self.topic_num, self.input_encoding_size)
        self.emb_weights = np.load('oemb_weights/resnet152.npz')
        topic_emb_weights = torch.from_numpy(self.emb_weights['ff_topic_W'] + self.emb_weights['ff_topic_b'])
        self.topic_embed.weight.data.copy_(topic_emb_weights)
        self.topic_embed.requires_grad = False
        self.img_embed.weight.data.copy_(torch.from_numpy(self.emb_weights['ff_image_W'].T))
        self.img_embed.bias.data.copy_(torch.from_numpy(self.emb_weights['ff_image_b']))
        self.img_embed.requires_grad = False

        self.img_topic_emb = nn.Linear(int(self.input_encoding_size * 2), self.input_encoding_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def forward(self, fc_feats, att_feats, topics, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt1 = self.img_embed(fc_feats)
                xt2 = self.topic_embed(topics.long())
                xt1_ = torch.norm(xt1, p=2, dim=1).detach()
                xt1 = xt1.div(xt1_.unsqueeze(dim=1).expand_as(xt1))
                xt1 = F.relu(xt1)
                
                xt2_ = torch.norm(xt2, p=2, dim=1).detach()
                xt2 = xt2.div(xt2_.unsqueeze(dim=1).expand_as(xt2))
                xt2 = F.relu(xt2)
                xt = xt1 + xt2
                #xt = self.img_topic_emb(torch.cat((xt1, xt2), 1))
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, i-1].clone()                
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].data.sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)
                
        output, state = self.core(xt.unsqueeze(0), state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, topics, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt1 = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                    xt2 = self.topic_embed(topics[k:k+1]).expand(beam_size, self.input_encoding_size)
                    xt1_ = torch.norm(xt1, p=2, dim=1).detach()
                    xt1 = xt1.div(xt1_.unsqueeze(dim=1).expand_as(xt1))
                    xt1 = F.relu(xt1)
                    
                    xt2_ = torch.norm(xt2, p=2, dim=1).detach()
                    xt2 = xt2.div(xt2_.unsqueeze(dim=1).expand_as(xt2))
                    xt2 = F.relu(xt2)
                    xt = xt1 + xt2
                    
                    #xt = self.img_topic_emb(torch.cat((xt1, xt2), 1))
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, topics, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, topics, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                xt1 = self.img_embed(fc_feats)
                xt2 = self.topic_embed(topics)
                xt1_ = torch.norm(xt1, p=2, dim=1).detach()
                xt1 = xt1.div(xt1_.unsqueeze(dim=1).expand_as(xt1))
                xt1 = F.relu(xt1)
                
                xt2_ = torch.norm(xt2, p=2, dim=1).detach()
                xt2 = xt2.div(xt2_.unsqueeze(dim=1).expand_as(xt2))
                xt2 = F.relu(xt2)
                xt = xt1 + xt2

                #xt = self.img_topic_emb(torch.cat((xt1, xt2), 1))
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                    it = it.view(-1).long() # and flatten indices for downstream processing

                xt = self.embed(Variable(it, requires_grad=False))

            if t >= 2:
                # stop when all finished
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)