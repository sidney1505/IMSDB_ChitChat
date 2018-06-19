from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time, git
import argparse
import os, code
import json
##code.interact(local=dict(globals(), **locals()))


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_from", default=None, help="specify path, where config of loaded model lies")
parser.add_argument("-s", "--skip_preprocessing", default=False, help="specify path, where config of loaded model lies")
parser.add_argument("-c", "--commit", default=False, help="specify path, where config of loaded model lies")

args = parser.parse_args()

# hi
"""Holds model hyperparams and data information."""
base_config = {
    'batch_size' : 100, # test
    'embed_size' : 80,
    'hidden_size' : 80,
    'current_epoch' : 0,
    'max_epochs' : 256,
    'early_stopping' : 200,
    'dropout' : 0.9,
    'lr' : 0.001,
    'l2' : 0.001,
    'cap_grads' : False,
    'max_grad_val' : 10,
    'noisy_grads' : False,
    'word2vec_init' : False,
    'embedding_init' : 1.732,
    'anneal_threshold' : 1000,
    'anneal_by' : 1.5,
    'num_hops' : 3,
    'num_attention_features' : 4,
    'max_allowed_inputs' : 130,
    'num_train' : 9000,
    'train_mode' : True,
    'dataset_location' : '/home/sbender/Documents/IMSDB/movie_scripts_simple_dataset_babiformat.json',
    'vocabulary_location' : '/home/sbender/Documents/IMSDB/simple_dataset_vocabs.txt',
    'max_num_train_data' : 1000,
    'max_num_test_data' : 1000,
    'preprocessed_dataset_location' : '/cvhci/users/sbender/gan_chitchat/data/movie_scripts_simple_dataset_babiformat.npz',
    'preprocess_data' : True,
    'global_step' : 0,
    'is_loaded' : False,
    'model_name' : None,
    'base_model_dir' : '/cvhci/users/sbender/gan_chitchat/models',
    'weights_path' : None,
    'log_dir' : None,
    'num_train' : None,
    'num_val' : None,
    'num_test' : None,
    'git_hash' : None
}

def writeLog(path, value):
    writer = open(path,'w')
    writer.write(str(value))
    writer.close()

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
# setup config
if args.load_from == None:
    config = base_config
    config['is_commited'] = args.commit
    config['model_name'] = str(time.strftime("%Y-%m-%d %H %M %S").replace(' ','_').replace('-','_'))
    if config['is_commited']:
        config['model_name'] = 'com' + config['model_name']
    os.makedirs(config['base_model_dir'] + '/' + config['model_name'])
    os.makedirs(config['base_model_dir'] + '/' + config['model_name'] + '/weights')
    config['log_dir'] = config['base_model_dir'] + '/' + config['model_name'] + '/logs'
    os.makedirs(config['log_dir'])
    config['train_accuracy_log_path'] = config['log_dir'] + '/train_accuracy_log.txt'
    config['train_loss_log_path'] = config['log_dir'] + '/train_loss_log.txt'
    config['val_accuracy_log_path'] = config['log_dir'] + '/val_accuracy_log.txt'
    config['val_loss_log_path'] = config['log_dir'] + '/val_loss_log.txt'
    config['config_path'] = config['base_model_dir'] + '/' + config['model_name'] + '/config.json'
    if config['is_commited']:
        os.system('git add *.py')
        os.system('git commit -m "model ' + config['model_name'] + ' started!"')
    repo = git.Repo(search_parent_directories=True)
    new_sha = repo.head.object.hexsha
    config['git_hash'] = new_sha
    config_string = json.dumps(config)
    config_writer = open(config['config_path'],'w')
    config_writer.write(config_string)
    config_writer.close()
else:
    config_reader = open(args.load_from,'r')
    config_string = config_reader.read()
    config_reader.close()
    config = json.loads(config_string)
    if config['is_commited'] and config['git_hash'] != sha:
        print('git hash doesnt match!')
        code.interact(local=dict(globals(), **locals()))

best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
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
            saver.restore(session, model.config['weights_path'])

        print('==> starting training')
        for epoch in range(config['max_epochs']):
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
            writeLog(config['train_accuracy_log_path'], train_accuracy)
            writeLog(config['train_loss_log_path'], train_loss)
            valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
            writeLog(config['val_accuracy_log_path'], val_accuracy)
            writeLog(config['val_loss_log_path'], val_loss)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, model.config['weights_path'])

            # anneal
            if train_loss > prev_epoch_loss * model.config['anneal_threshold']:
                model.config['lr'] /= model.config['anneal_by']
                print('annealed lr to %f' % model.config['lr'])

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config['early_stopping']:
                break
            print('Total time: {}'.format(time.time() - start))

        print('Best validation accuracy:', best_val_accuracy)
