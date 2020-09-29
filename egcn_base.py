import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Time': 2, '2_Group': 3, '2_Nation': 4,
                       '2_Indeterminate': 5, '2_Population_Center': 6, '2_Government': 7,
                       '2_Commercial': 8, '2_Non_Governmental': 9, '2_Media': 10, '2_Building_Grounds': 11,
                       '2_Numeric': 12, '2_State_or_Province': 13, '2_Region_General': 14, '2_Sports': 15,
                       '2_Crime': 16, '2_Land': 17, '2_Air': 18, '2_Water': 19, '2_Airport': 20,
                       '2_Sentence': 21, '2_Educational': 22, '2_Celestial': 23, '2_Underspecified': 24,
                       '2_Shooting': 25, '2_Special': 26, '2_Subarea_Facility': 27, '2_Path': 28,
                       '2_GPE_Cluster': 29, '2_Exploding': 30, '2_Water_Body': 31, '2_Land_Region_Natural': 32,
                       '2_Nuclear': 33, '2_Projectile': 34, '2_Region_International': 35, '2_Medical_Science': 36,
                       '2_Continent': 37, '2_Job_Title': 38, '2_County_or_District': 39, '2_Religious': 40,
                       '2_Contact_Info': 41, '2_Chemical': 42, '2_Subarea_Vehicle': 43, '2_Entertainment': 44,
                       '2_Biological': 45, '2_Boundary': 46, '2_Plant': 47, '2_Address': 48, '2_Sharp': 49,
                       '2_Blunt': 50
                       }

dep_dict = {'O': 0, 'punct': 1, 'iobj': 2, 'parataxis': 3, 'auxpass': 4, 'aux': 5,
            'conj': 6, 'advcl': 7, 'acl:relcl': 8, 'nsubjpass': 9, 'csubj': 10, 'compound': 11,
            'compound:prt': 12, 'mwe': 13, 'cop': 14, 'neg': 15, 'nmod:poss': 16, 'appos': 17,
            'cc:preconj': 18, 'nmod': 19, 'nsubj': 20, 'xcomp': 21, 'det:predet': 22,
            'nmod:npmod': 23, 'acl': 24, 'amod': 25, 'expl': 26, 'csubjpass': 27, 'case': 28,
            'ccomp': 29, 'dobj': 30, 'ROOT': 31, 'discourse': 32, 'nmod:tmod': 33, 'dep': 34,
            'nummod': 35, 'mark': 36, 'advmod': 37, 'cc': 38, 'det': 39
            }


class EDModel(nn.Module):

    def __init__(self, args, id_to_tag, device, pre_word_embed):
        super(EDModel, self).__init__()

        self.device = device
        self.gcn_model = EEGCN(device, pre_word_embed, args)
        self.gcn_dim = args.gcn_dim
        self.classifier = nn.Linear(self.gcn_dim, len(id_to_tag))

    def forward(self, word_sequence, x_len, entity_type_sequence, adj, dep):
        outputs, weight_adj = self.gcn_model(word_sequence, x_len, entity_type_sequence, adj, dep)
        logits = self.classifier(outputs)
        return logits, weight_adj


class EEGCN(nn.Module):
    def __init__(self, device, pre_word_embeds, args):
        super().__init__()

        self.device = device
        self.in_dim = args.word_embed_dim + args.bio_embed_dim
        self.maxLen = args.num_steps

        self.rnn_hidden = args.rnn_hidden
        self.rnn_dropout = args.rnn_dropout
        self.rnn_layers = args.rnn_layers

        self.gcn_dropout = args.gcn_dropout
        self.num_layers = args.num_layers
        self.gcn_dim = args.gcn_dim

        # Word Embedding Layer
        self.word_embed_dim = args.word_embed_dim
        self.wembeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pre_word_embeds), freeze=False)

        # Entity Label Embedding Layer
        self.bio_size = len(entity_subtype_dict)
        self.bio_embed_dim = args.bio_embed_dim
        if self.bio_embed_dim:
            self.bio_embeddings = nn.Embedding(num_embeddings=self.bio_size,
                                               embedding_dim=self.bio_embed_dim)

        self.dep_size = len(dep_dict)
        self.dep_embed_dim = args.dep_embed_dim
        self.edge_embeddings = nn.Embedding(num_embeddings=self.dep_size,
                                            embedding_dim=self.dep_embed_dim,
                                            padding_idx=0)

        self.rnn = nn.LSTM(self.in_dim, self.rnn_hidden, self.rnn_layers, batch_first=True, \
                           dropout=self.rnn_dropout, bidirectional=True)
        self.rnn_drop = nn.Dropout(self.rnn_dropout)  # use on last layer output

        self.input_W_G = nn.Linear(self.rnn_hidden * 2, self.gcn_dim)
        self.pooling = args.pooling
        self.gcn_layers = nn.ModuleList()
        self.gcn_drop = nn.Dropout(self.gcn_dropout)
        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(self.device, self.gcn_dim, self.dep_embed_dim, args.pooling))
        self.aggregate_W = nn.Linear(self.gcn_dim + self.num_layers * self.gcn_dim, self.gcn_dim)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.rnn_hidden, self.rnn_layers)
        h0, c0 = h0.to(self.device), c0.to(self.device)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, word_sequence, x_len, entity_type_sequence, adj, edge):

        BATCH_SIZE = word_sequence.shape[0]
        BATCH_MAX_LEN = x_len[0]  

        word_sequence = word_sequence[:, :BATCH_MAX_LEN].contiguous()
        adj = adj[:, :BATCH_MAX_LEN, :BATCH_MAX_LEN].contiguous()
        edge = edge[:, :BATCH_MAX_LEN, :BATCH_MAX_LEN].contiguous()
        weight_adj = self.edge_embeddings(edge)  # [batch, seq, seq, dim_e]

        word_emb = self.wembeddings(word_sequence)
        x_emb = word_emb
        if self.bio_embed_dim:
            entity_type_sequence = entity_type_sequence[:, :BATCH_MAX_LEN].contiguous()
            entity_label_emb = self.bio_embeddings(entity_type_sequence)
            x_emb = torch.cat([x_emb, entity_label_emb], dim=2)

        rnn_outputs = self.rnn_drop(self.encode_with_rnn(x_emb, x_len, BATCH_SIZE))
        gcn_inputs = self.input_W_G(rnn_outputs)
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        src_mask = (word_sequence != 0)
        src_mask = src_mask[:, :BATCH_MAX_LEN].unsqueeze(-2).contiguous()

        for _layer in range(self.num_layers):
            gcn_outputs, weight_adj = self.gcn_layers[_layer](weight_adj,gcn_outputs)  # [batch, seq, dim]
            gcn_outputs = self.gcn_drop(gcn_outputs)
            weight_adj = self.gcn_drop(weight_adj)
            layer_list.append(gcn_outputs)
        
        outputs = torch.cat(layer_list, dim=-1)
        aggregate_out = self.aggregate_W(outputs)
        return aggregate_out, weight_adj


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()

        self.gcn_dim = gcn_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling

        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = Edgeupdate(gcn_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, weight_adj, gcn_inputs):
        """
        :param weight_adj: [batch, seq, seq, dim_e]
        :param gcn_inputs: [batch, seq, dim]
        :return:
        """
        batch, seq, dim = gcn_inputs.shape
        weight_adj = weight_adj.permute(0, 3, 1, 2)  # [batch, dim_e, seq, seq]

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.dep_embed_dim, seq, dim)
        Ax = torch.matmul(weight_adj, gcn_inputs)  # [batch, dim_e, seq, dim]
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        # Edge update weight_adj[batch, dim_e, seq, seq]
        weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()  # [batch, seq, seq, dim_e]
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
        return node_outputs, edge_outputs


class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node: [batch, seq, seq, dim]
        :return:
        """

        node = torch.cat([node1, node2], dim=-1) # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


