# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import SGD

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from selfModules.neg import NEG_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--num-iterations', type=int, default=1000000, metavar='NI',
                        help='num iterations (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='BS',
                        help='batch size (default: 10)')
    parser.add_argument('--num-sample', type=int, default=5, metavar='NS',
                        help='num sample (default: 5)')
    parser.add_argument('--use-cuda', action='store_false',#default value True #, type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--path', type=str, default='', 
                        help='Path where the files are located')
    args = parser.parse_args()


    path=args.path

    data_files = [path + 'super/train_2.txt',
                       path + 'super/test_2.txt']

    idx_files = [path + 'super/words_vocab_2.pkl',
                      path + 'super/characters_vocab_2.pkl']

    tensor_files = [[path + 'super/train_word_tensor_2.npy',
                          path + 'super/valid_word_tensor_2.npy'],
                         [path + 'super/train_character_tensor_2.npy',
                          path + 'super/valid_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)




    # batch_loader_2 = BatchLoader('')
    params = Parameters(batch_loader_2.max_word_len,
                        batch_loader_2.max_seq_len,
                        batch_loader_2.words_vocab_size,
                        batch_loader_2.chars_vocab_size, path)

    neg_loss = NEG_loss(params.word_vocab_size, params.word_embed_size)
    if args.use_cuda:
        neg_loss = neg_loss.cuda()

    # NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
    optimizer = SGD(neg_loss.parameters(), 0.1)

    for iteration in range(args.num_iterations):

        input_idx, target_idx = batch_loader_2.next_embedding_seq(args.batch_size)

        input = Variable(t.from_numpy(input_idx).long())
        target = Variable(t.from_numpy(target_idx).long())
        if args.use_cuda:
            input, target = input.cuda(), target.cuda()

        out = neg_loss(input, target, args.num_sample).mean()

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        if iteration % 500 == 0:
            out = out.item()#.data[0]#.cpu().data.numpy()[0]
            print('iteration = {}, loss = {}'.format(iteration, out))

    word_embeddings = neg_loss.input_embeddings()
    #Saves the word embeddings at the end of this programs
    np.save(os.path.join(path,'super/word_embeddings.npy'), word_embeddings)
