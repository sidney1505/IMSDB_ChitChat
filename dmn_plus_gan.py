from __future__ import print_function
from __future__ import division

import sys
import time, code
###code.interact(local=dict(globals(), **locals()))

import numpy as np
from copy import deepcopy

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.layers import core as layers_core

import imsdb_input

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)

# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """We could have used RNN for parsing sentence but that tends to overfit.
    The simpler choice would be to take sum of embedding but we loose loose positional information.
    Position encoding is described in section 4.1 in "End to End Memory Networks" in more detail (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

class DMN_PLUS(object):

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        # set up embedding
        self.embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding") # TODO check if embedding is done correctly!
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()
        if self.config['is_loaded']:
            pass # TODO load data!


    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        # TODO self.vocabulary, self.batchsize, self.groundtruth
        if self.config['train_mode']:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.max_a_len = imsdb_input.load_imsdb(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.max_a_len = imsdb_input.load_imsdb(self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config['embed_size'])
        vocab_reader = open(self.config['vocabulary_location'],'r')
        self.vocabulary = vocab_reader.read().split('\n')
        vocab_reader.close()
        self.vocab_size = len(self.vocabulary) + 2 # UNK & EOS tokens -> +2
        self.batchsize = self.config['batch_size']

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config['batch_size'], self.max_q_len,))
        # TODO limit max_sentences to 20 and max_sen_len to 30
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config['batch_size'], self.max_sentences, self.max_sen_len,))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config['batch_size'],))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config['batch_size'],))

        self.answer_placeholder = tf.placeholder(tf.int32, shape=(self.config['batch_size'], self.max_a_len,))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, -1)
        return pred

    def add_loss_op(self, output):
        """Calculate loss"""
        # loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder))
        prediction_length = tf.argmax(tf.argmax(output, -1), -1)
        groundtruth_length = tf.argmax(self.answer_placeholder, -1)
        seq_length = tf.maximum(prediction_length, groundtruth_length)
        weights = tf.sequence_mask(seq_length, self.max_a_len, tf.float32)
        weights = tf.cast(weights, tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(output, self.answer_placeholder, weights) # TODO mask sequence

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config['l2']*tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config['lr'])
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config['cap_grads']:
            gvs = [(tf.clip_by_norm(grad, self.config['max_grad_val']), var) for grad, var in gvs]
        if self.config['noisy_grads']:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op
  

    def get_question_representation(self):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)

        gru_cell = tf.contrib.rnn.GRUCell(self.config['hidden_size'])
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                questions,
                dtype=np.float32,
                sequence_length=self.question_len_placeholder
        )

        return q_vec

    # input module
    def get_input_representation(self):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)
        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config['hidden_size'])
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config['hidden_size'])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_gru_cell,
                backward_gru_cell,
                inputs,
                dtype=np.float32,
                sequence_length=self.input_len_placeholder
        )

        # sum forward and backward output vectors
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):

            features = [fact_vec*q_vec,
                        fact_vec*prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                            self.config['embed_size'],
                            activation_fn=tf.nn.tanh,
                            reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                            1,
                            activation_fn=None,
                            reuse=reuse, scope="fc2")

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config['hidden_size']),
                    gru_inputs,
                    dtype=np.float32,
                    sequence_length=self.input_len_placeholder
            )

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""
        # TODO fit to sequence output
        '''rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)
        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                self.vocab_size,
                activation=None)'''
        projection_layer = layers_core.Dense(self.vocab_size, use_bias=False) # TODO self.vocabulary, self.batchsize, self.groundtruth
        embedded_size = self.vocab_size + 2 + 100 # because of start and end token
        GO_SYMBOL = self.vocab_size + 1
        END_SYMBOL = self.vocab_size
        start_tokens = tf.tile([GO_SYMBOL], [self.batchsize]) # TODO self.batchsize
        start_tokens2D = tf.expand_dims(start_tokens,1)
        def embedding(x):
            return tf.one_hot(x, embedded_size)
        # while training
        ##code.interact(local=dict(globals(), **locals()))
        nr_target = tf.cast(tf.ones([self.batchsize]) * tf.cast(self.max_a_len, tf.float32), tf.int32)
        decoder_hints = tf.concat([start_tokens2D, self.answer_placeholder], 1) # TODO self.groundtruth
        decoder_hints_embedded = embedding(decoder_hints)
        initial_state = tf.concat([rnn_output, q_vec],1)
        initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state, initial_state)
        decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(initial_state[0].shape[-1].value)
        def train_decode():
            train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_hints_embedded, nr_target)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_rnn_cell, train_helper, initial_state, output_layer=projection_layer)
            return tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_a_len)
        def infer_decode():
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens, END_SYMBOL)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_rnn_cell, infer_helper, initial_state, output_layer=projection_layer)
            return tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_a_len)
        decoder_output,_,_ = tf.cond(self.is_training, train_decode, infer_decode)
        self.energy = decoder_output[0]
        # code.interact(local=dict(globals(), **locals()))
        # TODO answers one hot encoden, sequence loss verwenden, use attention gru for decoding
        return self.energy

    def inference(self):
        """Performs inference on the DMN model"""

        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation()


        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation()

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config['num_hops']):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                            self.config['hidden_size'],
                            activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)

        return output


    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        losses = []
        accuracies = []

        # shuffle data
        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a = data
        qp, ip, ql, il, im, a = qp[p], ip[p], ql[p], il[p], im[p], a[p]

        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.dropout_placeholder: dp,
                  self.is_training: train}
            loss, pred, summary, _, energy = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op, self.energy], feed_dict=feed)
            
            '''feed2 = {self.test_output: pred,
                  self.answer_placeholder: a[index]}
            loss2 = session.run(self.test_loss, feed_dict=feed2)
            feed3 = {self.test_output_energy: energy,
                  self.answer_placeholder: a[index]}
            loss3 = session.run(self.test_loss2, feed_dict=feed3)
            code.interact(local=dict(globals(), **locals()))'''

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            answers = a[step*config.batch_size:(step+1)*config.batch_size]
            current_accuracy = self.calculateBatchAccuracy(pred, answers)
            accuracies.append(current_accuracy)
            losses.append(loss)

            if step % 5 == 0:
                print('Step:     ' + str(step))
                print('Loss:     ' + str(loss) + ' (' + str(np.mean(losses)) + ')')
                print('Accuracy: ' + str(current_accuracy) + ' (' + str(np.mean(accuracies)) + ')')
                print('Q:        ' + self.getSentenceString(qp[index][0]))
                print('GT:       ' + self.getSentenceString(answers[0]))
                print('Pred:     ' + self.getSentenceString(pred[0]))
                print('')
            '''if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(losses)))
                sys.stdout.flush()'''


        if verbose:
            sys.stdout.write('\r')

        return np.mean(losses), np.mean(accuracies)

    def getSentenceString(self, sentence):
        s = ''
        for it in range(len(sentence)):
            if sentence[it] < self.vocab_size - 2:
                s += self.vocabulary[sentence[it]] + ' '
            elif sentence[it] == self.vocab_size - 2:
                s += 'UNK' + ' ' # UNK token
            else:
                break # EOS token
        return s[:-1]

    def calculateBatchAccuracy(self, pred, gt):
        accuracies = []
        for batch in range(pred.shape[0]):
            accuracies.append(self.calculateSentenceAccuracy(pred[batch],gt[batch]))
        return np.mean(accuracies)

    def calculateSentenceAccuracy(self, pred, gt):
        num_correct = 0
        num_false = 0
        for word in range(pred.shape[0]):
            if pred[word] == self.vocab_size - 1:
                for word in range(word,pred.shape[0]):
                    if gt[word] == self.vocab_size - 1:
                        break
                    else:
                        num_false += 1
                break
            elif gt[word] == self.vocab_size - 1:
                for word in range(word,pred.shape[0]):
                    if pred[word] == self.vocab_size - 1:
                        break
                    else:
                        num_false += 1
                break
            else:
                if gt[word] == pred[word]:
                    num_correct += 1
                else:
                    num_false += 1
        return num_correct / float(num_correct + num_false)
