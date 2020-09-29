# encoding = utf8
import re, math, codecs, random
import numpy as np
from tqdm import trange

def iob2(tags):
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
    return True

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i , tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-','S-'))
        elif tag.split('-')[0] == 'I':
            if i+1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def create_dico(item_list):

    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):

    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i,v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def create_input(data):

    inputs = list()
    inputs.append(data['chars'])
    # inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embedding.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with'
          'pretrained embeddings.'% (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights

def get_doc_features(doc_id, char_to_id, doc_dict, chars):
    sentence_num = 8
    doc_sentence = doc_dict[doc_id[0]] 
    doc_chars = list()
    for sentence in doc_sentence:
        doc_char = [char_to_id[w if w in char_to_id else '<UNK>'] for w in sentence]
        doc_chars.append(doc_char)  
    a = doc_chars.index(chars) 
    if len(doc_chars) <= sentence_num:
        doc_chars = doc_chars
    else: #
        if a <= sentence_num/2:
            doc_chars = doc_chars[:sentence_num]
        elif len(doc_chars)-a <= sentence_num/4:
            doc_chars = doc_chars[-sentence_num:0]
        else:
            doc_chars = doc_chars[int(a-sentence_num/2):int(a+sentence_num/2)]
    return doc_chars


def get_dep_features(string, dep_rels):
    dep_dict = { 'O': 0, 'punct': 1, 'iobj': 2, 'parataxis': 3, 'auxpass': 4, 'aux': 5, 'conj': 6, 'advcl': 7, 'acl:relcl': 8, 'nsubjpass': 9,'csubj': 10, 'compound': 11, 'compound:prt': 12, 'mwe': 13, 'cop': 14, 'neg': 15, 'nmod:poss': 16, 'appos': 17, 'cc:preconj': 18, 'nmod': 19, 'nsubj': 20, 'xcomp': 21, 'det:predet': 22, 'nmod:npmod': 23, 'acl': 24, 'amod': 25, 'expl': 26, 'csubjpass': 27, 'case': 28, 'ccomp': 29, 'dobj': 30, 'ROOT': 31, 'discourse': 32, 'nmod:tmod': 33, 'dep': 34, 'nummod': 35, 'mark': 36, 'advmod': 37, 'cc': 38, 'det': 39}
    dep_features = list()
    for w in dep_rels:
        dep_feature = dep_dict[w]
        dep_features.append(dep_feature)
    return dep_features


def get_sub_features(string, entity_subtype):
    entity_subtype_dict = {'O': 0, '2_Individual': 1, '2_Time': 2, '2_Group': 3, '2_Nation': 4, '2_Indeterminate': 5, '2_Population_Center': 6, '2_Government': 7, '2_Commercial': 8, '2_Non_Governmental': 9, '2_Media': 10, '2_Building_Grounds': 11, '2_Numeric': 12, '2_State_or_Province': 13, '2_Region_General': 14, '2_Sports': 15, '2_Crime': 16, '2_Land': 17, '2_Air': 18, '2_Water': 19, '2_Airport': 20, '2_Sentence': 21, '2_Educational': 22, '2_Celestial': 23, '2_Underspecified': 24, '2_Shooting': 25, '2_Special': 26, '2_Subarea_Facility': 27, '2_Path': 28, '2_GPE_Cluster': 29, '2_Exploding': 30, '2_Water_Body': 31, '2_Land_Region_Natural': 32, '2_Nuclear': 33, '2_Projectile': 34, '2_Region_International': 35, '2_Medical_Science': 36, '2_Continent': 37, '2_Job_Title': 38, '2_County_or_District': 39, '2_Religious': 40, '2_Contact_Info': 41, '2_Chemical': 42, '2_Subarea_Vehicle': 43, '2_Entertainment': 44, '2_Biological': 45, '2_Boundary': 46, '2_Plant': 47, '2_Address': 48, '2_Sharp': 49, '2_Blunt': 50}
    subtype_featrues = list()
    for w in entity_subtype:
        if w == "O":
            subtype_featrue = 0
        else:
            subtype_featrue = entity_subtype_dict[w.split("-")[1]]
        subtype_featrues.append(subtype_featrue)
    return subtype_featrues


def get_seg_features(string,tags):
    tags_dict = {'O': 0, '1_PER': 1, '1_Time': 2, '1_GPE': 3, '1_ORG': 4, '1_FAC': 5, '1_LOC': 6, '1_VEH': 7, '1_Numeric': 8, '1_WEA': 9, '1_Crime': 10, '1_Sentence': 11, '1_Job_Title': 12, '1_Contact_Info': 13}
    seg_feature = []
    for tag in tags:
        if "1_PER" in tag:
            entity_tag = 1
        elif "1_GPE" in tag:
            entity_tag = 2
        elif "1_Time" in tag:
            entity_tag = 3
        elif "1_ORG" in tag:
            entity_tag = 4
        elif "1_FAC" in tag:
            entity_tag = 5
        elif "1_VEH" in tag:
            entity_tag = 6
        elif "1_GPE" in tag:
            entity_tag = 7
        elif "1_Numeric" in tag:
            entity_tag = 8
        elif "1_Crime" in tag:
            entity_tag = 9
        elif "1_Sentence" in tag:
            entity_tag = 10
        elif "1_Contact_Info" in tag:
            entity_tag = 11
        elif "1_Job_Title" in tag:
            entity_tag = 12
        elif "1_WEA" in tag:
            entity_tag = 13
        else:
            entity_tag = 0
        seg_feature.append(entity_tag)
    return seg_feature

class BatchManager(object):

    def __init__(self, data, batch_size, num_steps):
        # data: string, doc_chars, chars, types, subtypes, tags

        self.batch_data = self.sort_and_pad(data, batch_size, num_steps)
        self.len_data = len(self.batch_data)
        self.length = int(num_steps)
    def sort_and_pad(self, data, batch_size, num_steps):
        self.num_batch = int(math.ceil(len(data) / batch_size))
        print("num_batch: ", self.num_batch)
        lens = [len(x[0]) for x in data] # 句子长度

        sorted_data = sorted(data, key=lambda x:len(x[0]), reverse=True)
        batch_data = list()
        for i in trange(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size],num_steps))
        return batch_data

    @staticmethod
    def pad_data(data, length):

        strings = []
        chars = []
        segs = []
        subtypes = []
        targets = []
        adj, dep = [], []
        lens = []

        max_length = length
        for line in data:
            string, char, seg, subtype, target, dep_rel_features, dep_word_idx = line
            string_len = len(string)
            if string_len <= max_length:
                lens.append(string_len)
                padding = [0] * (max_length - len(string))
                strings.append(string + padding)
                chars.append(char + padding)
                segs.append(seg + padding)
                targets.append(target + padding)
                subtypes.append(subtype + padding)
            else:
                lens.append(max_length)
                strings.append(string[0:max_length])
                chars.append(char[0:max_length])
                targets.append(target[0:max_length])
                segs.append(seg[0:max_length])
                subtypes.append(subtype[0:max_length])

            # Dep:
            curr_adj = np.eye(max_length)
            curr_dep = np.random.randint(0, 1, (max_length, max_length), dtype=int)
            for j,dep_relation in enumerate(dep_rel_features):
                if j >= max_length:
                    break
                token1_id, token2_id = j, int(dep_word_idx[j])
                if token2_id == -1 or token2_id >= max_length:
                    token2_id = token1_id
                curr_adj[token1_id, token2_id], curr_adj[token2_id, token1_id] = 1, 1
                curr_dep[token1_id, token2_id], curr_dep[token2_id, token1_id] = int(dep_relation), int(dep_relation)
            adj.append(curr_adj)
            dep.append(curr_dep)
        return [strings, lens, chars, segs, subtypes, targets, adj, dep]

    def iter_batch(self, shuffle = False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def input_from_line(line, char_to_id):
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


