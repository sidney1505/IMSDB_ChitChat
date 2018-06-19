from __future__ import print_function
from __future__ import division

import tensorflow as tf

import time
import argparse
import os, code
import json
##code.interact(local=dict(globals(), **locals()))


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_from", type=string, default=None, help="specify path, where config of loaded model lies")
parser.add_argument("-sp", "--skip_preprocessing", default=False, help="specify path, where config of loaded model lies")

args = parser.parse_args()

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    current_epoch = 0
    max_epochs = 256
    early_stopping = 200

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = np.sqrt(3)

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    # ???
    max_allowed_inputs = 130
    num_train = 9000

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

    dataset_location = '/home/sbender/Documents/IMSDB/movie_scripts_simple_dataset_babiformat.json'
    vocabulary_location = '/home/sbender/Documents/IMSDB/simple_dataset_vocabs.txt'

    max_num_train_data = 1000000
    max_num_test_data = 1000000

    preprocessed_dataset_location = '/cvhci/users/sbender/gan_chitchat/data/movie_scripts_simple_dataset_babiformat.npz'
    preprocess_data = True

    global_step = 0
    is_loaded = False
    model_name = None
    base_model_dir = '/cvhci/users/sbender/gan_chitchat/models'
    weights_path = base_model_dir + '/weights'
    log_dir = base_model_dir + '/logs'
    num_train = None
    num_val = None
    num_test = None

def writeLog(path, value):
    writer = open(path,'w')
    writer.write(str(value))
    writer.close()

if args.load_from != None:
    config = Config()
    config.model_name = time.strftime("%Y-%m-%d %H %M").replace(' ','_').replace('-','_')
    os.makedirs(config.base_model_dir + '/' + config.model_name)
    os.makedirs(config.base_model_dir + '/' + config.model_name + '/weights')
    config.log_dir = config.base_model_dir + '/' + config.model_name + '/logs'
    os.makedirs(config.log_dir)
    config.train_accuracy_log_path = config.log_dir + '/train_accuracy_log.txt'
    config.train_loss_log_path = config.log_dir + '/train_loss_log.txt'
    config.al_accuracy_log_path = config.log_dir + '/val_accuracy_log.txt'
    config.val_loss_log_path = config.log_dir + '/val_loss_log.txt'
    config.config_path = config.base_model_dir + '/' + config.model_name + '/config.json'
    config_string = json.dump(config)
    config_writer = open(config.config_path,'w')
    config_writer.write(config_string)
    config_writer.close()
else:
    config_reader = open(args.load_from,'r')
    config_string = config_reader.read()
    config_reader.close()
    config = json.loads(config_string)

best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "plus":
        from dmn_plus_gan import DMN_PLUS
        model = DMN_PLUS(config)

for run in range(num_runs):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        if args.restore:
            print('==> restoring weights')
            saver.restore(session, model.config.weights_path)

        print('==> starting training')
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            #code.interact(local=dict(globals(), **locals()))
            train_loss, train_accuracy = model.run_epoch(
                session,
                model.train,
                epoch,
                train_writer,
                train_op=model.train_step,
                train=True)
            writeLog(config.train_accuracy_log_path, train_accuracy)
            writeLog(config.train_loss_log_path, train_loss)
            valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
            writeLog(config.val_accuracy_log_path, val_accuracy)
            writeLog(config.val_loss_log_path, val_loss)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, model.config.weights_path)

            # anneal
            if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                model.config.lr /= model.config.anneal_by
                print('annealed lr to %f' % model.config.lr)

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

        print('Best validation accuracy:', best_val_accuracy)
