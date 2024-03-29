from __future__ import division
from __future__ import print_function

import sys, code, json
####code.interact(local=dict(globals(), **locals()))

import os as os
import numpy as np

# can be sentence or word
input_mask_mode = "sentence"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init_babi(fname=None):    
    print("==> Loading test from %s" % fname)
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""} 
            counter = 0
            id_map = {}
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line
            id_map[id] = counter
            counter += 1
            
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = []
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print("==> loading glove")
    with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print("==> glove is loaded")
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if to_return == "one_hot":
        if word in vocab:
            one_hot = vocab.index(word)
        else:
            one_hot = len(vocab)
        return one_hot

    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    else:
        return -1

def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False, max_num_data=1000000):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    num_fails = 0
    num_sucesses = 0
    for x in data_raw:
        try:
            # process inputs
            if split_sentences:
                inp = x["C"].lower().split(' . ') 
                inp = [w for w in inp if len(w) > 0]
                inp = [i.split() for i in inp]
            else:
                inp = x["C"].lower().split(' ') 
                inp = [w for w in inp if len(w) > 0]
            if split_sentences: 
                inp_vector = [[process_word(word = w, 
                                            word2vec = word2vec, 
                                            vocab = vocab, 
                                            ivocab = ivocab, 
                                            word_vector_size = embed_size, 
                                            to_return = "one_hot") for w in s] for s in inp]
            else:
                inp_vector = [process_word(word = w, 
                                            word2vec = word2vec, 
                                            vocab = vocab, 
                                            ivocab = ivocab, 
                                            word_vector_size = embed_size, 
                                            to_return = "one_hot") for w in inp]
            if split_sentences:
                inputs.append(inp_vector)
            else:
                inputs.append(np.vstack(inp_vector).astype(floatX))
            # process questions
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            q_vector = [process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "one_hot") for w in q]
            questions.append(np.vstack(q_vector).astype(floatX)) # what is the problem with the question vectors???
            # process answers
            a = x["A"].lower().split(' ')
            a = [w for w in a if len(w) > 0]
            a_vector = [process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "one_hot") for w in a]
            answers.append(np.vstack(a_vector).astype(floatX))
            # NOTE: here we assume the answer is one word! 
            '''answers.append(process_word(word = x["A"], 
                                            word2vec = word2vec, 
                                            vocab = vocab, 
                                            ivocab = ivocab, 
                                            word_vector_size = embed_size, 
                                            to_return = "index"))'''

            if not split_sentences:
                if input_mask_mode == 'word':
                    input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
                elif input_mask_mode == 'sentence': 
                    input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
                else:
                    raise Exception("invalid input_mask_mode")

            num_sucesses += 1
            # #code.interact(local=dict(globals(), **locals()))
            if num_sucesses % 100 == 0:
                print(str(num_sucesses) + " / " + str(len(data_raw)))
                if num_sucesses >= max_num_data:
                    break #'''
        except Exception as e:
            # reset the lists
            print(e)
            min_length = min(min(len(questions),len(inputs)),len(answers))
            questions = questions[:min_length]
            inputs = inputs[:min_length]
            answers = answers[:min_length]
            num_fails += 1
    print('num_fails: ' + str(num_fails) + ' num_sucesses: ' + str(len(inputs)))
    assert len(inputs) == len(questions) == len(answers), "length isnt the same! " + str(len(inputs)) + ', '  + str(len(questions)) + ', ' + str(len(answers)) + ', '

    return inputs, questions, answers, input_masks

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    # necessary to normalize model
    MAX_LENS = 30
    MAX_SEN_LEN = 30
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = min(len(s),MAX_SEN_LEN)
        lens[i] = min(len(t),MAX_LENS)
        sen_lens.append(sentence_lens)
        if sentence_lens.shape[0] != 0:
            max_sen_lens.append(np.max(sentence_lens))
        else:
            max_sen_lens.append(0)
    return lens, sen_lens, max(max_sen_lens)
    
# grouping by size???
def pad_inputs(inputs, num_sentences=None, max_num_sentences=None, mode="", sen_lens=None, max_sen_len=None, vocab_size=None):
    print('pad inputs!')
    print(mode)
    if mode == "mask":
        padded = [np.pad(inp, (0, max_num_sentences - num_sentences[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)
    # restrict max sentences and sentence lengths
    elif mode == "split_sentences":
        padded = np.ones((len(inputs), max_num_sentences, max_sen_len)) * (vocab_size - 1)
        num_words = 0
        for i, inp in enumerate(inputs):
            #code.interact(local=dict(globals(), **locals()))
            for j in range(max_num_sentences):
                for k in range(max_sen_len):
                    if j < len(inp) and k < len(inp[j]):
                        padded[i][j][k] = inp[j][k]
            if i % 100 == 0:
                print(str(i))
            '''# dirty guranties, that shape is correct
            padded_sentences = [np.pad(s[:max_sen_len], (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp[max(0,len(inp)-max_num_sentences):])]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_num_sentences:
                padded_sentences = padded_sentences[(len(padded_sentences)-max_num_sentences):]
                lens[i] = max_num_sentences
            try:
                if len(sen_lens[i]) > 0:
                    padded_sentences = np.vstack(padded_sentences)
                    padded_sentences = np.pad(padded_sentences, ((0, max_num_sentences - lens[i]),(0,0)), 'constant', constant_values=0)
                    padded[i] = padded_sentences
                num_sucesses += 1
                if num_sucesses % 100 == 0:
                    print(str(num_sucesses) + ' / ' + str(len(inputs)))
            except Exception:
                if num_fails <= 3:
                    #code.interact(local=dict(globals(), **locals()))
                    pass
                num_fails += 1'''
        # print('num_fails: ' + str(num_fails) + ' num_sucesses: ' + str(num_sucesses))
        return padded
    else:
        try:
            #padded = [np.pad(np.squeeze(inp, axis=1), (0, max(0,max_len - lens[i])), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
            padded = [np.expand_dims(np.pad(inp, ((0, max(0,max_sen_len - sen_lens[i])), (0,0)), 'constant', constant_values=vocab_size-1),0) for i, inp in enumerate(inputs)]
            #padded = np.expand_dims(padded, 0)
            return np.vstack(padded)
        except Exception as e:
            print(e)
            #code.interact(local=dict(globals(), **locals()))

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

# returns self.train, self.valid, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size
# self.train = questions[:config['num_train']], inputs[:config['num_train']], q_lens[:config['num_train']], input_lens[:config['num_train']], input_masks[:config['num_train']], answers[:config['num_train']]
#{'Q': 'Where is the football', 'A': 'garden', 'C': 'Mary moved to the bathroom . Sandra journeyed to the bedroom . Mary got the football there . John went to the kitchen . Mary went back to the kitchen . Mary went back to the garden . ', 'S': [2, 5]}
def load_imsdb(config, split_sentences=True):
    #
    #split_sentences = False
    if not config['skip_preprocessing']:
        dataset_reader = open(config['dataset_location'],'r')
        imsdb_data = json.load(dataset_reader, strict=False)
        test_border = int(len(imsdb_data) * 0.9)
        babi_train_raw = imsdb_data[:test_border]
        babi_test_raw = imsdb_data[test_border:]
        dataset_reader.close()
        ###
        # vocab = {}
        vocab_reader = open(config['vocabulary_location'],'r')
        vocab = vocab_reader.read()
        vocab_reader.close()
        vocab = vocab.split('\n')
        vocab_size = len(vocab) + 2 # UNK & EOS
        # no idea what this is for!!!
        ivocab = {}

        #babi_train_raw_old, babi_test_raw_old = get_babi_raw(config['babi_id'], config['babi_test_id'])

        if config['word2vec_init']:
            assert config['embed_size'] == 100
            word2vec = load_glove(config['embed_size'])
        else:
            word2vec = {}

        # set word at index zero to be end of sentence token so padding with zeros is consistent
        '''process_word(word = "<eos>", \
                    word2vec = word2vec, \
                    vocab = vocab, \
                    ivocab = ivocab, \
                    word_vector_size = config['embed_size'], \
                    to_return = "index")'''

        print('==> get train inputs')
        train_data = process_input(babi_train_raw, np.float32, word2vec, vocab, ivocab, config['embed_size'], split_sentences, config['max_num_train_data'])
        print('==> get test inputs')
        test_data = process_input(babi_test_raw, np.float32, word2vec, vocab, ivocab, config['embed_size'], split_sentences, config['max_num_test_data'])

        if config['word2vec_init']:
            assert config['embed_size'] == 100
            word_embedding = create_embedding(word2vec, ivocab, config['embed_size'])
        else:
            word_embedding = np.random.uniform(-config['embedding_init'], config['embedding_init'], (len(ivocab), config['embed_size'])) # TODO what is done here and why does it work???
            # word_embedding = np.random.normal(size=[vocab_size, config['embed_size']]) 

        inputs, questions, answers, input_masks = train_data if config['train_mode'] else test_data
        if split_sentences:
            input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
            max_mask_len = max_sen_len
        else:
            input_lens = get_lens(inputs)
            mask_lens = get_lens(input_masks)
            max_mask_len = np.max(mask_lens)
        max_input_len = min(np.max(input_lens), config['max_allowed_input_length'])
        #pad out arrays to max
        if split_sentences:
            inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len, vocab_size=len(vocab)+2).astype(int)
            input_masks = np.zeros(len(inputs))
        else:
            inputs = pad_inputs(inputs, input_lens, max_input_len)
            input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

        q_lens = get_lens(questions)
        max_q_len = np.max(q_lens)
        questions = pad_inputs(questions, sen_lens=q_lens, max_sen_len=max_q_len, vocab_size=len(vocab)+2).astype(int)

        a_lens = get_lens(answers)
        max_a_len = np.max(a_lens)
        answers = pad_inputs(answers, sen_lens=a_lens, max_sen_len=max_a_len, vocab_size=len(vocab)+2).astype(int)

        answers = np.squeeze(answers)
        questions = np.squeeze(questions)
        with open(config['preprocessed_dataset_location'], 'w') as fout:
            np.savez(fout, \
                inputs=inputs, \
                questions=questions, \
                answers=answers, \
                input_masks=input_masks, \
                input_lens=input_lens, \
                q_lens=q_lens, \
                a_lens=a_lens, \
                word_embedding=word_embedding)
    else:
        data = np.load(config['preprocessed_dataset_location'])
        inputs = data['inputs']
        questions = data['questions']
        answers = data['answers']
        input_masks = data['input_masks']
        q_lens = data['q_lens']
        a_lens = data['a_lens']
        input_lens = data['input_lens']
        word_embedding = data['word_embedding']
        max_q_len = np.max(q_lens)
        max_a_len = np.max(a_lens)
        max_input_len = min(np.max(input_lens), config['max_allowed_input_length'])
        max_mask_len = inputs.shape[2] # ???
    #
    print('max_a_len')
    print(max_a_len)
    print('max_q_len')
    print(max_q_len)
    #
    if config['train_mode']:
        config['num_train'] = int(answers.shape[0] * 0.9)
        config['num_val'] = int(answers.shape[0] - config['num_train'])
        train = questions[:config['num_train']], inputs[:config['num_train']], q_lens[:config['num_train']], input_lens[:config['num_train']], input_masks[:config['num_train']], answers[:config['num_train']]
        valid = questions[config['num_train']:], inputs[config['num_train']:], q_lens[config['num_train']:], input_lens[config['num_train']:], input_masks[config['num_train']:], answers[config['num_train']:]
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, max_a_len

    else:
        config['num_test'] = int(answers.shape[0])
        config['preprocess_data'] = False
        test = questions, inputs, q_lens, input_lens, input_masks, answers
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, max_a_len
