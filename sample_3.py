import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE
from torch.autograd import Variable

if __name__ == '__main__':

    assert os.path.exists('trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num-sample', type=int, default=5, metavar='NS',
                        help='num samplings (default: 5)')
    parser.add_argument('--num-sentence', type=int, default=10, metavar='NS',
                        help='num samplings (default: 10)')
    args = parser.parse_args()
    
    file_1 = open('test.txt', 'r')
    data = file_1.readlines()

    file_2 = open('test_2.txt', 'r')
    data_2 = file_2.readlines()    

    path=''
    
    ''' ============================= BatchLoader loading ===============================================
    '''
    data_files = [path + 'data/train.txt',
                       path + 'data/test.txt']

    idx_files = [path + 'data/words_vocab.pkl',
                      path + 'data/characters_vocab.pkl']

    tensor_files = [[path + 'data/train_word_tensor.npy',
                          path + 'data/valid_word_tensor.npy'],
                         [path + 'data/train_character_tensor.npy',
                          path + 'data/valid_character_tensor.npy']]

    batch_loader = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    ''' ============================= BatchLoader loading ===============================================
    '''

    data_files = [path + 'data/super/train_2.txt',
                       path + 'data/super/test_2.txt']

    idx_files = [path + 'data/super/words_vocab_2.pkl',
                      path + 'data/super/characters_vocab_2.pkl']

    tensor_files = [[path + 'data/super/train_word_tensor_2.npy',
                          path + 'data/super/valid_word_tensor_2.npy'],
                         [path + 'data/super/train_character_tensor_2.npy',
                          path + 'data/super/valid_character_tensor_2.npy']]

    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                            batch_loader_2.max_seq_len,
                            batch_loader_2.words_vocab_size,
                            batch_loader_2.chars_vocab_size)

    '''======================================== RVAE creation ==================================================
    '''
    
    rvae = RVAE(parameters,parameters_2)
    rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    n_best = 3 
    beam_size=10 
    
    assert n_best <= beam_size 

    for i in range(args.num_sentence):

        '''================================================== Input Encoder-1 ========================================================
        '''
        use_cuda = 1
        input = batch_loader.next_batch(1, 'valid', i)
        input = [Variable(t.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]

        [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input


        ''' =================================================== Input for Encoder-2 ========================================================
        '''

        input_2 = batch_loader_2.next_batch(1, 'valid', i)
        input_2 = [Variable(t.from_numpy(var)) for var in input_2]
        input_2 = [var.long() for var in input_2]
        input_2 = [var.cuda() if use_cuda else var for var in input_2]

        [encoder_word_input_2, encoder_character_input_2, decoder_word_input_2, decoder_character_input_2, target] = input_2

        ''' ================================================== Forward pass ===========================================================
        '''
        # exit()

        logits,_,kld,mu,std = rvae.forward(0.,
                              encoder_word_input, encoder_character_input,
                              encoder_word_input_2,encoder_character_input_2,
                              decoder_word_input_2, decoder_character_input_2,
                              z=None)

        ''' ================================================================================================================================
        '''

        # print '============'
        print (data[i])
        print (data_2[i])
        # print '------------------------------------'




        for iteration in range(args.num_sample):
            # seed = np.random.normal(size=[1, parameters.latent_variable_size])
            seed = Variable(t.randn([1, parameters.latent_variable_size]))
            # seed = Variable(t.from_numpy(seed).float())
            # exit()
            # seed = mu
            # if use_cuda:
            seed = seed.cuda()

            seed = seed * std + mu
            # seed = seed*std + mu
            # print 'Multiplication done'
            # seed = seed.cuda()
            # print seed.size
            # print type(seed)
            # print seed
            # exit()
            results, scores = rvae.sampler(batch_loader,batch_loader_2, 50, seed, args.use_cuda,i,beam_size,n_best)
            # exit()
            # print(results)
            for tt in results:
                for k in xrange(n_best):
                    sen = " ". join([batch_loader_2.decode_word(x[k]) for x in tt])
                # print sen
                if batch_loader.end_token in sen:    
                    print sen[:sen.index(batch_loader.end_token)]
                else :
                    print sen
            # exit()       
        print '\n'


    # print 'words_vocab_size BatchLoader ----------->'
    # print batch_loader.words_vocab_size
    # print '-----------------------------------------'

    # print 'words_vocab_size BatchLoader_2 ----------->'
    # print batch_loader_2.words_vocab_size
    # print '-----------------------------------------'

