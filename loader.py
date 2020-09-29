import os
import re
import codecs

from data_utils import iob2, iob_iobes, create_dico, create_mapping, get_seg_features, get_sub_features, get_dep_features

def load_sentences(path):
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line =line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence) 
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word= line.split()
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences, tag_scheme):
    for i,s in enumerate(sentences):
        # tags = [w[-1] for w in s]
        tags = [w[4] for w in s]
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format!'
                            + 'please check sentence %i:\n%s' %(i,s_str))
        if tag_scheme == 'iob':
            for word, new_tag in zip(s, tags):
                word[4] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[4] = new_tag
        else:
            raise Exception('Unknow tagging scheme!')

def char_mapping(sentences, lower):
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char

def augment_with_pretrained(dictionary,ext_emb_path,chars):
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
    ])
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id,id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def tag_mapping(sentences):
    tags = [[char[-1] for char in s ] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def prepare_dataset(sentences, char_to_id, tag_to_id, train = True):

    none_index = tag_to_id['O']
    data = []

    for s in sentences: 
        string, entity_types, entity_subtype, tags, dep_rels, dep_word_idx = list(), list(), list(), list(), list(), list()
        for w in s:
            if w[0] != "...":
                string.append(w[0]) # token --> sentence
                entity_types.append(w[2])
                entity_subtype.append(w[3])
                tags.append(w[4])
                dep_rels.append(w[5])
                dep_word_idx.append(w[-1])
        if len(string)> 4: 
            chars = [char_to_id[w if w in char_to_id else '<UNK>']
                     for w in string] 
            types = get_seg_features(string, entity_types)  # convert to id
            subtypes = get_sub_features(string, entity_subtype) # convert to id
            dep_rel_features = get_dep_features(string, dep_rels)
            if train:
                tags = [tag_to_id[w] for w in tags]
            else:
                tags = [none_index for _ in chars]
            data.append([string, chars, types, subtypes, tags, dep_rel_features, dep_word_idx])
    return data



