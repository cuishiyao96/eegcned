# Edge-Enhanced Graph Convolution Networks for Event Detection with Syntactic Relation

This code is for **Findings of EMNLP2020** paper **Edge-Enhanced Graph Convolution Networks for Event Detection with Syntactic Relation**. 

Event detection (ED), a key subtask of information extraction, aims to recognize instances of specific event types in text. Previous studies on the task have verified the effectiveness of integrating syntactic dependency into graph convolutional networks. However, these methods usually ignore dependency label information, which conveys rich and useful linguistic knowledge for ED. In this paper, we propose a novel architecture named Edge-Enhanced Graph Convolution Networks (EE-GCN), which simultaneously exploits syntactic structure and typed dependency label information to perform ED. Specifically, an edge-aware node update module is designed to generate expressive word representations by aggregating syntactically-connected words through specific dependency types. Furthermore, to fully explore clues hidden in dependency edges, a node-aware edge update module is introduced, which refines the relation representations with contextual information. These two modules are complementary to each other and work in a mutual promotion way. We conduct experiments on the widely used ACE2005 dataset and the results show significant improvement over competitive baseline methods:

![Performance of EEGCN](https://github.com/cuishiyao96/eegcned/blob/master/fig/performance.png?raw=true)

You can find the paper [here](https://www.aclweb.org/anthology/2020.findings-emnlp.211/)


See below for an overview of the model architecture:

![Overview of our architecture](https://github.com/cuishiyao96/eegcned/blob/master/fig/model.png?raw=true)

## Requirements
- python 3.7.4
- [Pytorch==1.1.0](https://pytorch.org/)

## Datasets
The ACE 2005 dataset can be download from：https://catalog.ldc.upenn.edu/LDC2006T06

## Train and test the model

* Train:

python main.py --log_name test --cuda 0 --epoch 100 --weight_decay 0.00001 --label_weights 5 --optimizer SGD --lr 0.1 --bio_embed_dim 25 --dep_embed_dim 50 --num_steps 50 --rnn_dropout 0.6 --gcn_dropout 0.6 --train True --batch_size 30

* Test:

python main.py --log_name model --cuda 0 --epoch 100 --weight_decay 0.00001 --label_weights 5 --optimizer SGD --lr 0.1 --bio_embed_dim 25 --dep_embed_dim 50 --num_steps 50 --rnn_dropout 0.6 --gcn_dropout 0.6 --train False --batch_size 30

## Related Repo


Data pre-processing and evaluation functions are adapted from [NBTNGMA4ED](https://github.com/yubochen/NBTNGMA4ED/).

## Usage

### Important files
- ./data_doc: example_data
- 100.utf8: pretrained word embedding
- maps.pkl: the map of word and word_id, event labels and label_id;

### Data format
The training, dev and test data is expected in standard tab-separated format. One word per line, separate column for token and label, empty line between sentences.
* The first column is assumed to be the token.
* The second column is the current document ID
* The third column is the type of the entity. 
* The fourth column is the entity subtyp of the entity.
* The fifth column is the event-type label.
* The sixth column is the syntactic dependency label.
* The last column is index of the syntactic connected token.
For example:
```
it CNN_ENG_20030612_173004.10 O O O nsubj 4
is CNN_ENG_20030612_173004.10 O O O cop 4
， CNN_ENG_20030612_173004.10 O O O advmod 4
in CNN_ENG_20030612_173004.10 O O O case 4
fact CNN_ENG_20030612_173004.10 O O O ROOT -1
， CNN_ENG_20030612_173004.10 O O O case 8
the CNN_ENG_20030612_173004.10 O O O det 8
deadliest CNN_ENG_20030612_173004.10 O O O amod 8
conflict CNN_ENG_20030612_173004.10 O O B-Life_Die nmod 4
since CNN_ENG_20030612_173004.10 O O O mark 12
world CNN_ENG_20030612_173004.10 O O B-Conflict_Attack compound 11
war CNN_ENG_20030612_173004.10 O O I-Conflict_Attack nsubj 12
ii CNN_ENG_20030612_173004.10 O O I-Conflict_Attack dep 8
and CNN_ENG_20030612_173004.10 O O O cc 12
has CNN_ENG_20030612_173004.10 O O O aux 16
been CNN_ENG_20030612_173004.10 O O O aux 16
going CNN_ENG_20030612_173004.10 O O O conj 12
on CNN_ENG_20030612_173004.10 O O O compound:prt 16
for CNN_ENG_20030612_173004.10 O O O case 20
five CNN_ENG_20030612_173004.10 B-1_Time B-2_Time O nummod 20
years CNN_ENG_20030612_173004.10 I-1_Time I-2_Time O nmod 16
. CNN_ENG_20030612_173004.10 O O O punct 4
```

## Citation
```
@inproceedings{cui-etal-2020-edge,
    title = "Edge-Enhanced Graph Convolution Networks for Event Detection with Syntactic Relation",
    author = "Cui, Shiyao  and
      Yu, Bowen  and
      Liu, Tingwen  and
      Zhang, Zhenyu  and
      Wang, Xuebin  and
      Shi, Jinqiao",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.211",
    pages = "2329--2339",
    abstract = "Event detection (ED), a key subtask of information extraction, aims to recognize instances of specific event types in text. Previous studies on the task have verified the effectiveness of integrating syntactic dependency into graph convolutional networks. However, these methods usually ignore dependency label information, which conveys rich and useful linguistic knowledge for ED. In this paper, we propose a novel architecture named Edge-Enhanced Graph Convolution Networks (EE-GCN), which simultaneously exploits syntactic structure and typed dependency label information to perform ED. Specifically, an edge-aware node update module is designed to generate expressive word representations by aggregating syntactically-connected words through specific dependency types. Furthermore, to fully explore clues hidden from dependency edges, a node-aware edge update module is introduced, which refines the relation representations with contextual information.These two modules are complementary to each other and work in a mutual promotion way. We conduct experiments on the widely used ACE2005 dataset and the results show significant improvement over competitive baseline methods.",
}
```




