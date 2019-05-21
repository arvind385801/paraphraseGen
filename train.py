import argparse
import os
import csv
import numpy as np
import datetime
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

def writeColumns(filename, zipList):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        for row in zipList:
            writer.writerow(row) # for more writerow
    return

if __name__ == "__main__":

    folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', action='store_false',#default value True #type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=str, default='', metavar='UT',
                        help='load pretrained model path to it')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')
    parser.add_argument('--path', type=str, default='', 
                        help='Path where the files are located')
    args = parser.parse_args()
    path=args.path
    os.makedirs(os.path.join(path,folder))

    if not os.path.exists(path+'word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    
    ''' =================== Creating batch_loader for encoder-1 =========================================
    '''
    data_files = [path + 'train.txt',
                       path + 'test.txt']

    idx_files = [path + 'words_vocab.pkl',
                      path + 'characters_vocab.pkl']

    tensor_files = [[path + 'train_word_tensor.npy',
                          path + 'valid_word_tensor.npy'],
                         [path + 'train_character_tensor.npy',
                          path + 'valid_character_tensor.npy']]

    batch_loader = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size,
                            path)


    ''' =================== Doing the same for encoder-2 ===============================================
    '''
    data_files = [path + 'super/train_2.txt',
                       path + 'super/test_2.txt']

    idx_files = [path + 'super/words_vocab_2.pkl',
                      path + 'super/characters_vocab_2.pkl']

    tensor_files = [[path + 'super/train_word_tensor_2.npy',
                          path + 'super/valid_word_tensor_2.npy'],
                         [path + 'super/train_character_tensor_2.npy',
                          path + 'super/valid_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                            batch_loader_2.max_seq_len,
                            batch_loader_2.words_vocab_size,
                            batch_loader_2.chars_vocab_size,
                            path)
    '''=================================================================================================
    '''


    rvae = RVAE(parameters,parameters_2)
    if args.use_trained != '':
        trainedModelName = os.path.join(os.path.join(args.use_trained,'trained_RVAE'))
        rvae.load_state_dict(t.load(trainedModelName))
    if args.use_cuda:
        print ("Using cuda")
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer,batch_loader, batch_loader_2)
    validate = rvae.validater(batch_loader,batch_loader_2)

    loss_tr_result = ["loss_train"]
    ce_result = ["cross_entropy_train"]
    kld_result = ["kld_train"]
    coef_result = ["coef_train"]
    it = ["iteration"]
    loss_val_result =["loss_val"]

    start_index = 0
    # start_index_2 = 0

    trainSize = len(batch_loader.word_tensor[0]) # 0 train - 1 validation
    limitBatch = int(trainSize/args.batch_size)*args.batch_size

    print trainSize

    for iteration in range(args.num_iterations):
        #This needs to be changed
        #start_index =  (start_index+1)%50000 quora # 331164 coco
        start_index = (start_index+args.batch_size)%limitBatch#331136#149163 # coco 331136/331164 # quora 49984/50000
        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout, start_index)

        # exit()

        if iteration % 10 == 0:
            cross_entropy = round(cross_entropy.item(),2)
            kld = round(kld.item(),2)
            coef = round(coef,5)
            print('\n')
            #print('------------TRAIN-------------')
            print('ITERATION\tCROSS_ENT\tKLD\tCOEF')
            print(iteration,"  ",cross_entropy,"  ",kld,"  ",coef) #.cpu().numpy()[0]

            if iteration %10 ==0:
                it.append(iteration)
                ce_result.append(cross_entropy)
                kld_result.append(kld)
                coef_result.append(coef)

        # if iteration % 10 == 0:
        #     start_index_2 = (start_index_2+args.batch_size)%3900
        #     cross_entropy, kld = validate(args.batch_size, args.use_cuda, start_index_2)

        #     cross_entropy = cross_entropy.data.cpu().numpy()[0]
        #     kld = kld.data.cpu().numpy()[0]

        #     print('\n')
        #     print('------------VALID-------------')
        #     print('--------CROSS-ENTROPY---------')
        #     print(cross_entropy)
        #     print('-------------KLD--------------')
        #     print(kld)
        #     print('------------------------------')

        #     ce_result += [cross_entropy]
        #     kld_result += [kld]
            '''
         if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader_2, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(sample)
            print('------------------------------')
            '''
    t.save(rvae.state_dict(), 'trained_RVAE')
    writeColumns(os.path.join(path,folder,"log_loss.csv"), zip(it,ce_result,kld_result,coef_result))
    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
