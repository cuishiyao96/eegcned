# encoding = utf-8
import os
import pickle
import torch
import argparse
import random
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from egcn_base import EDModel as Model
from logger import Log
from loader import load_sentences, update_tag_scheme
from loader import prepare_dataset
from utils import new_masked_cross_entropy, adjust_learning_rate, get_learning_rate
from utils import test_ner, str2bool
from data_utils import iobes_iob, BatchManager, load_word2vec


def main():
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_sentences = load_sentences(args.train_file)
    dev_sentences = load_sentences(args.dev_file)
    test_sentences = load_sentences(args.test_file)

    update_tag_scheme(train_sentences, args.tag_schema)
    update_tag_scheme(test_sentences, args.tag_schema)
    update_tag_scheme(dev_sentences, args.tag_schema)

    with open(args.map_file, 'rb') as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id
    )

    train_manager = BatchManager(train_data, args.batch_size, args.num_steps)
    dev_manager = BatchManager(dev_data, 100, args.num_steps)
    test_manager = BatchManager(test_data, 100, args.num_steps)

    if args.cuda >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(args.cuda)
    else:
        device = torch.device('cpu')
    print("device: ", device)

    if args.train:
        train(id_to_char, id_to_tag, train_manager, dev_manager, device)
    f1, res_info = eval_model(id_to_char, id_to_tag, test_manager, device, args.log_name)
    log_handler.info("\n resinfo {} \v F1: {} ".format(res_info, f1))


def eval_model(id_to_char, id_to_tag, test_manager, device, model_name=None):
    print("Eval ......")
    if not model_name:
        model_name = args.log_name
    old_weights = np.random.rand(len(id_to_char), args.word_embed_dim)
    pre_word_embed = load_word2vec("100.utf8", id_to_char, args.word_embed_dim, old_weights)
    e_model = Model(args, id_to_tag, device, pre_word_embed).to(device)
    e_model.load_state_dict(torch.load("./models/" + model_name + ".pkl"))
    print("model loaded ...")

    e_model.eval()
    all_results = []
    for batch in test_manager.iter_batch():

        strs, lens, chars, segs, subtypes, tags, adj, dep = batch
        chars = torch.LongTensor(chars).to(device)
        _lens = torch.LongTensor(lens).to(device)
        subtypes = torch.LongTensor(subtypes).to(device)
        tags = torch.LongTensor(tags).to(device)
        adj = torch.FloatTensor(adj).to(device)
        dep = torch.LongTensor(dep).to(device)
        logits,_ = e_model(chars, _lens, subtypes, adj, dep)

        """ Evaluate """
        # Decode
        batch_paths = []
        for index in range(len(logits)):
            length = lens[index]
            score = logits[index][:length]  # [seq, dim]
            probs = F.softmax(score, dim=-1)  # [seq, dim]
            path = torch.argmax(probs, dim=-1)  # [seq]
            batch_paths.append(path)

        for i in range(len(strs)):
            result = []
            string = strs[i][:lens[i]]
            gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lens[i]]])
            pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lens[i]]])
            for char, gold, pred in zip(string, gold, pred):
                result.append(" ".join([char, gold, pred]))
            all_results.append(result)

    all_eval_lines = test_ner(all_results, args.result_path, args.log_name)
    res_info = all_eval_lines[1].strip()
    f1 = float(res_info.split()[-1])
    print("eval: f1: {}".format(f1))
    return f1, res_info


def train(id_to_char, id_to_tag, train_manager, dev_manager, device):
    old_weights = np.random.rand(len(id_to_char), args.word_embed_dim)
    pre_word_embed = load_word2vec("100.utf8", id_to_char, args.word_embed_dim, old_weights)

    if args.label_weights:
        label_weights = torch.ones([len(id_to_tag)]) * args.label_weights
        label_weights[0] = 1.0  # none
        label_weights = label_weights.to(device)
    else:
        label_weights = None

    model = Model(args, id_to_tag, device, pre_word_embed).to(device)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("device: ", model.device)
    MAX_F1 = 0

    for epoch in range(args.epoch):

        log_handler.info("Epoch: {} / {} ：".format(epoch + 1, args.epoch))
        log_handler.info("epoch {}, lr: {} ".format(epoch + 1, get_learning_rate(optimizer)))

        loss = train_epoch(model, optimizer, train_manager, label_weights, device)
        log_handler.info("epoch {}, loss : {}".format(epoch + 1, loss))
        f1, dev_model = dev_epoch(epoch, model, dev_manager, id_to_tag, device)
        log_handler.info("epoch {}, f1 : {}".format(epoch + 1, f1))
        if f1 > MAX_F1:
            MAX_F1 = f1
            torch.save(dev_model.state_dict(), "./models/{}.pkl".format(args.log_name))
        log_handler.info("epoch {}, MAX_F1: {}\n".format(epoch + 1, MAX_F1))
        print()


def dev_epoch(epoch, model, dev_manager, id_to_tag, device):
    # dev
    model.eval()
    all_results = []
    for batch in dev_manager.iter_batch():

        strs, lens, chars, segs, subtypes, tags, adj, dep = batch
        chars = torch.LongTensor(chars).to(device)
        _lens = torch.LongTensor(lens).to(device)
        subtypes = torch.LongTensor(subtypes).to(device)
        tags = torch.LongTensor(tags).to(device)
        adj = torch.FloatTensor(adj).to(device)
        dep = torch.LongTensor(dep).to(device)
        logits,_ = model(chars, _lens, subtypes, adj, dep)  # [batch, seq, dim]

        """ Evaluate """
        # Decode
        batch_paths = []
        for index in range(len(logits)):
            length = lens[index]
            score = logits[index][:length]  # [seq, dim]
            probs = F.softmax(score, dim=-1)  # [seq, dim]
            path = torch.argmax(probs, dim=-1)  # [seq]
            batch_paths.append(path)

        for i in range(len(strs)):
            result = []
            string = strs[i][:lens[i]]
            gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lens[i]]])
            pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lens[i]]])
            for char, gold, pred in zip(string, gold, pred):
                result.append(" ".join([char, gold, pred]))
            all_results.append(result)

    all_eval_lines = test_ner(all_results, args.result_path, args.log_name)
    log_handler.info("epoch: {}, info: {}".format(epoch + 1, all_eval_lines[1].strip()))
    f1 = float(all_eval_lines[1].strip().split()[-1])
    return f1, model


def train_epoch(model, optimizer, train_manager, label_weights, device):
    total_loss = 0
    model.train()
    for i, batch in enumerate(tqdm(train_manager.iter_batch(shuffle=True))):
        optimizer.zero_grad()
        strs, lens, chars, segs, subtypes, tags, adj, dep = batch
        chars = torch.LongTensor(chars).to(device)
        lens = torch.LongTensor(lens).to(device)
        subtypes = torch.LongTensor(subtypes).to(device)
        tags = torch.LongTensor(tags).to(device)
        adj = torch.FloatTensor(adj).to(device)
        dep = torch.LongTensor(dep).to(device)
        # print('dep: ', dep.shape, dep.sum())

        logits,_ = model(chars, lens, subtypes, adj, dep)
        loss = new_masked_cross_entropy(logits, tags, lens, device, label_weights=label_weights)
        total_loss = total_loss + loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / train_manager.num_batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='egnn for ed')

    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--tag_schema', default="iob", type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_sharp_decay', default=0, type=int)
    parser.add_argument('--lr_min', default=0.00001, type=float)
    parser.add_argument('--label_weights', default=5, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str, help='Adam;SGD')
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    parser.add_argument('--num_steps', default=50, type=int)
    parser.add_argument('--word_embed_dim', default=100, type=int)
    parser.add_argument('--bio_embed_dim', default=25, type=int)
    parser.add_argument('--pos_embed_dim', default=0, type=int)
    parser.add_argument('--position_embed_dim', default=0, type=int)
    parser.add_argument('--dep_embed_dim', default=25, type=int)
    parser.add_argument('--rnn_hidden', default=100, type=int)
    parser.add_argument('--rnn_layers', default=1, type=int)
    parser.add_argument('--rnn_dropout', default=0.5, type=float)

    parser.add_argument('--pooling', default='avg', type=str)
    parser.add_argument('--gcn_dim', default=150, type=int)
    parser.add_argument('--num_layers', default=2, type=int, help='num of AGGCN layer blocks')
    parser.add_argument('--gcn_dropout', default=0.5, type=float)

    parser.add_argument('--map_file', default='maps.pkl', type=str)
    parser.add_argument('--result_path', default='result', type=str)
    parser.add_argument('--emb_file', default='100.utf8', type=str)
    parser.add_argument('--train_file', default=os.path.join("data_doc", "train"))
    parser.add_argument('--dev_file', default=os.path.join("data_doc", "dev"))
    parser.add_argument('--test_file', default=os.path.join("data_doc", "test"))
    parser.add_argument('--log_name', default='test', type=str)
    parser.add_argument('--seed', default=1023, type=int)

    args = parser.parse_args()
    log = Log(args.log_name + ".log")
    log_handler = log.getLog()

    log_handler.info("\nArgs: ")
    for arg in vars(args):
        log_handler.info("{}: {}".format(arg, getattr(args, arg)))
    log_handler.info("\n")

    main()
