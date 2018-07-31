import argparse
import os
import time

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.tensor import preprocess_data
from utils.parameters import Parameters
from model.rvae import RVAE
from torch.autograd import Variable
from six.moves import cPickle

if __name__ == '__main__':

    assert os.path.exists('./trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num-sample', type=int, default=5, metavar='NS',
                        help='num samplings (default: 5)')
    parser.add_argument('--num-sentence', type=int, default=10, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--beam-top', type=int, default=3, metavar='NS',
                        help='beam top (default: 1)')
    parser.add_argument('--beam-size', type=int, default=10, metavar='NS',
                        help='beam size (default: 10)')
    parser.add_argument('--use-file', type=bool, default=True, metavar='NS',
                        help='use file (default: False)')
    #Path to test file ---
    parser.add_argument('--test-file', type=str, default='data/test.txt', metavar='NS',
                        help='test file path (default: data/test.txt)')
    parser.add_argument('--save-model', type=str, default='./trained_RVAE', metavar='NS',
                        help='trained model save path (default: ./trained_models/trained_RVAE_quora)')
    args = parser.parse_args()
    
    #Removing, is already some previous files exist from last execution of program
    if os.path.exists('data/test_word_tensor.npy'):
        os.remove('data/test_word_tensor.npy')
    if os.path.exists('data/test_character_tensor.npy'):
        os.remove('data/test_character_tensor.npy')

    str =''
    if not args.use_file:
        str = raw_input("Input Question : ")
    else:
        file_1 = open(args.test_file, 'r')
        data = file_1.readlines()

    ''' ================================= BatchLoader loading ===============================================
    '''
    data_files = [args.test_file]

    idx_files = ['data/words_vocab.pkl',
                      'data/characters_vocab.pkl']

    tensor_files = [['data/test_word_tensor.npy'],
                         ['data/test_character_tensor.npy']]

    preprocess_data(data_files, idx_files, tensor_files, args.use_file, str)

    batch_loader = BatchLoader(data_files, idx_files, tensor_files)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)


    ''' ============================ BatchLoader for Question-2 ===============================================
    '''
    data_files = ['data/super/train_2.txt']

    idx_files = ['data/super/words_vocab_2.pkl',
                      'data/super/characters_vocab_2.pkl']

    tensor_files = [['data/super/train_word_tensor_2.npy'],
                         ['data/super/train_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                            batch_loader_2.max_seq_len,
                            batch_loader_2.words_vocab_size,
                            batch_loader_2.chars_vocab_size)


    '''======================================== RVAE loading ==================================================
    '''
    print 'Started loading'
    start_time = time.time()
    rvae = RVAE(parameters,parameters_2)
    rvae.load_state_dict(t.load(args.save_model))
    if args.use_cuda:
        rvae = rvae.cuda()
    loading_time=time.time() - start_time
    print 'Time elapsed in loading model =' , loading_time
    print 'Finished loading'

    ''' ==================================== Parameters Initialising ===========================================
    '''
    n_best = args.beam_top 
    beam_size =args.beam_size 
    
    assert n_best <= beam_size 
    use_cuda = args.use_cuda

    if args.use_file:
        num_sentence = args.num_sentence
    else:
        num_sentence = 1

    ''' =======================================================================================================
    '''

    for i in range(len(data)):
        if args.use_file:
            print (data[i])
        else:
            print str + '\n'
        for iteration in range(args.num_sample):

            seed = Variable(t.randn([1, parameters.latent_variable_size]))
            seed = seed.cuda()

            results, scores = rvae.sampler(batch_loader,batch_loader_2, 50, seed, args.use_cuda,i,beam_size,n_best)

            for tt in results:
                for k in xrange(n_best):
                    sen = " ". join([batch_loader_2.decode_word(x[k]) for x in tt])
                if batch_loader.end_token in sen:    
                    print sen[:sen.index(batch_loader.end_token)]
                else :
                    print sen      
        print '\n'