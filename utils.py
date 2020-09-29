import os
from conlleval import return_report


model_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_cript = os.path.join(eval_path, "conlleval")

import torch
from torch.autograd import Variable
from torch.nn import functional as F

def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def get_learning_rate(optimizer):
    lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
    assert len(lr_list) == 1
    return lr_list[0]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v == 'True':
        return True
    elif v == 'False':
        return False

def sequence_mask(sequence_length, device, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).to(device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def new_masked_cross_entropy(logits, target, length, device, label_weights=None):
    # length = Variable(torch.LongTensor(length)).cuda()
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    batch, max_batch_length, dim = logits.shape

    mask = sequence_mask(sequence_length=length, device=device, max_len=max_batch_length)  # mask: (batch, max_len)
    target = target[:, :max_batch_length].contiguous()
    target = target.masked_select(mask.to(device)) # [N]

    mask = mask.view(batch, max_batch_length, -1).contiguous() # [batch, len, 1]
    logits = logits.masked_select(mask.to(device))
    logits = logits.view(-1, dim) 

    loss = F.cross_entropy(logits, target, weight=label_weights)
    return loss


def test_ner(results, path, filename):
    output_file = os.path.join(path, filename)
    with open(output_file, "w", encoding='utf-8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item
