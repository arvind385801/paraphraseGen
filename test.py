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
    parser.add_argument('--test-file', type=str, default='test.txt', metavar='NS',
                        help='test file path (default: test.txt)')
    parser.add_argument('--save-model', type=str, default='./trained_RVAE', metavar='NS',
                        help='trained model save path (default: ./trained_models/trained_RVAE_quora)')
    parser.add_argument('--path', type=str, default='', metavar='PA',
                        help='path where the input files are')
    parser.add_argument('--modelPath', type=str, default='', metavar='PA',
                        help='path where the MODEL is saved')
    parser.add_argument('--testSize', type=int, default=0, metavar='NS',
                        help='beam size (default: 10)')


    args = parser.parse_args()
    
    print args.save_model
    assert os.path.exists(args.save_model), \
        'trained model not found'

    #Removing, is already some previous files exist from last execution of program
    if os.path.exists('test_word_tensor.npy'):
        os.remove('test_word_tensor.npy')
    if os.path.exists('test_character_tensor.npy'):
        os.remove('test_character_tensor.npy')

    str =''
    if not args.use_file:
        str = raw_input("Input Question : ")
    else:
        file_1 = open(os.path.join(args.path,args.test_file), 'r')
        data = file_1.readlines()

    ''' ================================= BatchLoader loading ===============================================
    '''
    data_files = [args.path+args.test_file]

    idx_files = [args.path+'words_vocab.pkl',
                      args.path+'characters_vocab.pkl']

    tensor_files = [[args.path+'test_word_tensor.npy'],
                         [args.path+'test_character_tensor.npy']]

    preprocess_data(data_files, idx_files, tensor_files, args.use_file, str)

    batch_loader = BatchLoader(data_files, idx_files, tensor_files)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size, args.path)


    ''' ============================ BatchLoader for Question-2 ===============================================
    '''
    data_files = [args.path+'super/train_2.txt']

    idx_files = [args.path+'super/words_vocab_2.pkl',
                      args.path+'super/characters_vocab_2.pkl']

    tensor_files = [[args.path+'super/train_word_tensor_2.npy'],
                         [args.path+'super/train_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                            batch_loader_2.max_seq_len,
                            batch_loader_2.words_vocab_size,
                            batch_loader_2.chars_vocab_size, args.path)

    '''======================================== RVAE loading ==================================================
    '''
    print 'Started loading'
    start_time = time.time()
    rvae = RVAE(parameters,parameters_2)
    #rvae.load_state_dict(t.load(os.path.join(os.path.join(args.modelPath,'trained_RVAE'))))
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

    fileOpt1 = open(args.path+"hyps.opt1","w") 
    fileOpt2 = open(args.path+"hyps.opt2","w") 
    fileOpt3 = open(args.path+"hyps.opt3","w") 
    fileOpt4 = open(args.path+"hyps.opt4","w") 
    fileOpt5 = open(args.path+"hyps.opt5","w") 

    fileList = [fileOpt1,fileOpt2,fileOpt3,fileOpt4,fileOpt5]

    if args.testSize == 0:
        sizeSamples = len(data)
    else:
        sizeSamples = args.testSize
    for i in range(len(data[:sizeSamples])):
        if args.use_file:
            print (i,data[i])
        else:
            print str + '\n'
        for iteration in range(args.num_sample):

            seed = Variable(t.randn([1, parameters.latent_variable_size]))
            seed = seed.cuda()

            results, scores = rvae.sampler(batch_loader,batch_loader_2, 50, seed, args.use_cuda,i,beam_size,n_best)

            for tt in results:
                for k in xrange(n_best):
                    sen = " ". join([batch_loader_2.decode_word(x[k]) for x in tt]) #adding lower
                if batch_loader.end_token in sen:    
                    senAux = sen[:sen.index(batch_loader.end_token)]
                else:
                    senAux = sen
                print senAux     
            fileList[iteration].write(senAux.lower()+'\n')
        print '\n'
