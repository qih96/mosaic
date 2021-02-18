import argparse
from solvers import train_init,train, train_distill
from gaussian_uniform.weighted_pseudo_list import make_weighted_pseudo_list
import copy
import torch
import os
import shutil
import time

def main(args):
    args.log_file.write('\n\n###########  initialization ############')
    
    # initializing
    acc, model = train_init(args)

    # acc = 0
    # model = torch.load(os.path.join(args.save_dir, 'initial_model.pk'))
    
    best_acc = acc
    best_model = copy.deepcopy(model)

    for stage in range(args.stages):
        
        print('\n\n########### stage : {:d}th ##############\n\n'.format(stage))
        args.log_file.write('\n\n########### stage : {:d}th    ##############'.format(stage))
        
        #updating parameters of gaussian-uniform mixture model with fixed network parameters，the updated pseudo labels and 
        #posterior probability of correct labeling is listed in folder "./data/office(dataset name)/pseudo_list"
        # make_weighted_pseudo_list(args, model)
        
        #updating network parameters with fixed gussian-uniform mixture model and pseudo labels
        acc,model = train_distill(best_model, args)
        # acc,model = train(args)
        
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            
    # torch.save(best_model,'snapshot/save/final_best_model.pk')
    print('final_best_acc:{:.4f}'.format(best_acc))
    return best_acc,best_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset',type=str,default='office')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target',type=str,default='dslr')
    parser.add_argument('--source_list', type=str, default='data/office/amazon_list.txt', help="The source dataset path list")
    parser.add_argument('--target_list', type=str, default='data/office/dslr_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=100, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--num_class',type=int,default=31,help='the number of classes')
    parser.add_argument('--stages',type=int,default=0,help='the number of alternative iteration stages')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--batch_size',type=int,default=36)
    parser.add_argument('--mosaic_1',type=int,default=1)
    parser.add_argument('--mosaic_2',type=int,default=1)
    parser.add_argument('--log_file')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    save_dir = args.dataset + '_' + args.baseline + '_'+ args.output_dir
    save_dir = 'snapshot/{}/{}'.format(args.dataset, save_dir)
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open('{}/log.txt'.format(save_dir),'w')
    log_file.write('dataset:{}\tsource:{}\ttarget:{}\n\n'
                   ''.format(args.dataset,args.source,args.target))
    args.log_file = log_file
    
    
    args.save_dir = save_dir
    print(args.save_dir)
    txt_dict = {'webcam': './data/list/office/webcam_list.txt', 
                'amazon': './data/list/office/amazon_list.txt', 
                'dslr':'./data/list/office/dslr_list.txt', 
                'train':'./data/list/visda-2017/train_list.txt', 
                'validation':'./data/list/visda-2017/validation_list.txt', 
                'art':'./data/list/office-home/Art.txt',
                'clipart':'./data/list/office-home/Clipart.txt', 
                'product':'./data/list/office-home/Product.txt', 
                'real_world':'./data/list/office-home/Real_World.txt',
                'i': './data/list/image-clef/i_list.txt',
                'p': './data/list/image-clef/p_list.txt',
                'c': './data/list/image-clef/c_list.txt',
            }
    num_class_dict = {'office-home':65, 'office':31,'visda2017':12, 'image-clef':12}

    args.num_class = num_class_dict[args.dataset]
    args.source_list = txt_dict[args.source]
    args.target_list = txt_dict[args.target]
    print('source dataset : {}'.format(args.source_list))
    print('target dataset : {}'.format(args.target_list))

    with open('{}/args.txt'.format(save_dir),'w') as f:
        for eachArg, value in args.__dict__.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    main(args)
