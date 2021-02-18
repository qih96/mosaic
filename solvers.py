import torch
import network
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import lr_schedule
import copy
import os
import utils
import torch.nn.functional as F

import contextlib
import tsne
import numpy as np
import itertools

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):

            m.track_running_stats ^= True
            m.training ^= True


    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open('data/{}/pseudo_list/{}_{}_list.txt'
                                     ''.format(args.dataset,args.source,args.target)).readlines(),
                                transform=image_train(),pseudo=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class).cuda()
    adv_net = network.AdversarialNetwork(in_feature=model.output_num(),hidden_size=1024,max_iter=2000).cuda()
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2] + adv_net.get_parameters()
    optimizer_classifier = torch.optim.SGD(parameter_classifier,lr=args.lr,momentum=0.9,weight_decay=0.005)
    optimizer_feature = torch.optim.SGD(parameter_feature,lr=args.lr,momentum=0.9,weight_decay=0)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\n iter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/best_model.pk')

        model.train(True)
        adv_net.train(True)
        optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier,i)
        optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        inputs_source, labels_source = inputs_source.cuda(),  labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        adv_loss = utils.loss_adv(features,adv_net)
        H = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))
        target_robust_loss = utils.robust_pseudo_loss(outputs_target,pseudo_labels_target,weights)

        classifier_loss = source_class_loss + target_robust_loss
        optimizer_classifier.zero_grad()
        classifier_loss.backward(retain_graph=True)
        optimizer_classifier.step()

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i,max_iter=2000)
        elif args.baseline =='DANN':
            lam = 0.0
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        feature_loss = classifier_loss + adv_loss + lam*loss_sm + lam*H
        optimizer_feature.zero_grad()
        feature_loss.backward()
        optimizer_feature.step()

        print('step:{: d},\t,source_class_loss:{:.4f},\t,target_robust_loss:{:.4f}'
              ''.format(i, source_class_loss.item(),target_robust_loss.item()))

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model


def train_init(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=2, drop_last=True)
    dsets["target"] = ImageList(open(args.target_list).readlines(), \
                                transform=image_train(), params=args)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=2, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=2)

    #model
    model = network.ResNet(class_num=args.num_class).cuda()
    adv_net = network.AdversarialNetwork(in_feature=model.output_num(),hidden_size=1024,max_iter=args.max_iter).cuda()
    parameter_list = model.get_parameters() + adv_net.get_parameters()
    optimizer = torch.optim.SGD(parameter_list,lr=args.lr,momentum=0.9,weight_decay=0.005)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)
    print_interval = (args.test_interval // 10)
    nt_cent = utils.NTXentLoss('cuda', args.batch_size, 0.2, True)

    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    max_batch = 100
    queue_size = args.batch_size * max_batch
    queue_data = [torch.zeros(queue_size, 256).cuda(), torch.zeros(queue_size, args.num_class).cuda()]
    queue_data_w = [torch.zeros(queue_size, 256).cuda(), torch.zeros(queue_size, args.num_class).cuda()]
    # queue_data = [torch.zeros(queue_size, 256).cuda(), torch.zeros(queue_size, 256).cuda()]

    queue_labels = [torch.ones(queue_size).cuda() * (args.num_class+1), torch.ones(queue_size).cuda() * (args.num_class+1)]
    queue_ptr = torch.zeros(1, dtype=torch.long)

    queue_weight = np.power(np.linspace(.0, 1.0, max_batch), 3)

    queue_weight = np.repeat(queue_weight, args.batch_size)


    # ema = utils.ModelEMA(model)
    # best_ema_acc = 0.0
    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\niter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)

            # ema.ema.train(False)
            # temp_acc = image_classification_test(dset_loaders, ema.ema)
            # if temp_acc > best_ema_acc:
            #     best_ema_acc = temp_acc
            #     # best_model = copy.deepcopy(model)
            # log_str = "\niter: {:05d}, \t precision: {:.4f},\t best_ema_acc:{:.4f}".format(i, temp_acc, best_ema_acc)
            # args.log_file.write(log_str)
            # args.log_file.flush()
            # print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(best_model,os.path.join(args.save_dir, 'initial_model.pk'))

        model.train(True)
        adv_net.train(True)
        optimizer = lr_schedule.inv_lr_scheduler(optimizer,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, inputs_target_2, inputs_target_mosaic_w, inputs_target_mosaic_s, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs_target_mosaic_w, inputs_target_mosaic_s =  inputs_target_mosaic_w.cuda(), inputs_target_mosaic_s.cuda()
        inputs_target_2 = inputs_target_2.cuda()
        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        # classifier_loss = utils.cross_entropy_with_logits(outputs_source, torch.eye(args.num_class)[labels_source].cuda(), 0.05)
        adv_loss = utils.loss_adv(features,adv_net)

        H = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i)
        elif args.baseline =='DANN':
            lam = 0.0
        prob_max, pseu_labels_target = torch.max(F.softmax(outputs_target, dim=1), dim=1)
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        # total_loss = classifier_loss + adv_loss + lam*loss_sm + network.calc_coeff(max(i-300, 0), high=0.01)*H
        total_loss = classifier_loss + adv_loss + lam*loss_sm + network.calc_coeff(max(i-300, 0), high=0.1)*H
        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mosaic_loss_target = torch.zeros(1)

        with _disable_tracking_bn_stats(model):
            mosaic_features_target_w, mosaic_outputs_target_w = model(inputs_target_mosaic_w)
            mosaic_features_target_s, mosaic_outputs_target_s = model(inputs_target_mosaic_s)
            with torch.no_grad():
                features_list_w = [mosaic_features_target_w, F.softmax(mosaic_outputs_target_w, dim=1)]

                features_target, outputs_target = model(inputs_target)
                # features_target_2, outputs_target_2 = model(inputs_target_2)
                # features_target = (features_target_1 + features_target_2) / 2.0
                # features_target = features_target.detach()
                # features_target_, outputs_target_ = ema.ema(inputs_target)
                p = (torch.softmax(outputs_target, dim=1) + torch.softmax(outputs_target, dim=1)) / 2
                pt = p**(1/1.0)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
                prob_max, pseu_labels_target = torch.max(targets_u, dim=1)
                features_list = [features_target, targets_u]
                labels_list = [pseu_labels_target, pseu_labels_target]

                utils.rightshift(queue_weight, args.batch_size)
                for j in range(len(features_list)):
                    queue_data[j][queue_ptr:queue_ptr+args.batch_size, :] = features_list[j]
                    queue_data_w[j][queue_ptr:queue_ptr+args.batch_size, :] = features_list_w[j]
                    queue_labels[j][queue_ptr:queue_ptr+args.batch_size] = labels_list[j]
                pre_ptr = int(queue_ptr)
                ptr = ((i+1) % max_batch) * args.batch_size
                queue_ptr[0] = ptr

            # mosaic_loss_target = (1.0*nt_cent(queue_data[0].detach(), mosaic_features_target_w, queue_labels[0], 
            #     pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False) +
            #                         1.0*nt_cent(queue_data_w[0].detach(), mosaic_features_target_s, queue_labels[0], 
            #     pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False)) * (network.calc_coeff(max(i, 0), high=0.3, low=0.01,
            #          max_iter=args.max_iter//5*2))

            mosaic_loss_target = (1.0*nt_cent(queue_data[1].detach(), F.softmax(mosaic_outputs_target_w, dim=1), queue_labels[1], 
                pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False) +
                                    1.0*nt_cent(queue_data_w[1].detach(), F.softmax(mosaic_outputs_target_s, dim=1), queue_labels[1], 
                pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False)) * (network.calc_coeff(max(i-max_batch, 0), high=0.3, low=0.01,
                        max_iter=args.max_iter//5*2))
            # mosaic_loss_target = 0.6*(torch.mean(torch.sum(torch.abs(targets_u - F.softmax(mosaic_outputs_target_w, dim=1)), dim=1)) +\
            #                      torch.mean(torch.sum(torch.abs(F.softmax(mosaic_outputs_target_w, dim=1).detach() - F.softmax(mosaic_outputs_target_s, dim=1)), dim=1)))

            mosaic_loss = mosaic_loss_target * 1.0

            mosaic_loss.backward()
            optimizer.step()

        # ema.update(model)

        if i % print_interval == 0:
            log_str = 'step:{: d},\t,class_loss:{:.4f},\t,adv_loss:{:.4f}\t,mosaic_loss:{:.4f}\t,mean_prob:{:.4f}'.format(i, classifier_loss.item(),
                                                        adv_loss.item(), mosaic_loss_target.item(),prob_max.mean().item())
            print(log_str)
            args.log_file.write('\n'+log_str)
            args.log_file.flush()

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model



def train_distill(teacher, args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=2, drop_last=True)
    dsets["target"] = ImageList(open(args.target_list).readlines(), \
                                transform=image_train(), num_patch_1=args.mosaic_1, num_patch_2=args.mosaic_2)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=2, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=2)

    #model
    model = network.ResNet(class_num=args.num_class).cuda()
    adv_net = network.AdversarialNetwork(in_feature=model.output_num(),hidden_size=1024).cuda()
    parameter_list = model.get_parameters() + adv_net.get_parameters()
    optimizer = torch.optim.SGD(parameter_list,lr=args.lr,momentum=0.9,weight_decay=0.005)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)
    print_interval = (args.test_interval // 10)
    nt_cent = utils.NTXentLoss('cuda', args.batch_size, 0.2, True)
    
    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    max_batch = 100
    queue_size = args.batch_size * max_batch
    queue_data = [torch.randn(queue_size, 256).cuda(), torch.randn(queue_size, args.num_class).cuda()]
    queue_data_w = [torch.randn(queue_size, 256).cuda(), torch.randn(queue_size, args.num_class).cuda()]
    # queue_data = [torch.randn(queue_size, 256).cuda(), torch.randn(queue_size, 256).cuda()]

    queue_labels = [torch.ones(queue_size).cuda() * (args.num_class+1), torch.ones(queue_size).cuda() * (args.num_class+1)]
    queue_ptr = torch.zeros(1, dtype=torch.long)

    queue_weight = np.power(np.linspace(.0, 1.0, max_batch), 3)

    queue_weight = np.repeat(queue_weight, args.batch_size)


    best_ema_acc = 0.0
    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\niter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)

            
            temp_acc = image_classification_test(dset_loaders, teacher)
            if temp_acc > best_ema_acc:
                best_ema_acc = temp_acc
                # best_model = copy.deepcopy(model)
            log_str = "\niter: {:05d}, \t precision: {:.4f},\t best_ema_acc:{:.4f}".format(i, temp_acc, best_ema_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        # if i % args.snapshot_interval == args.snapshot_interval -1:
        #     if not os.path.exists(args.save_dir):
        #         os.mkdir(args.save_dir)
        #     torch.save(best_model,os.path.join(args.save_dir, 'initial_model.pk'))

        model.train(True)
        adv_net.train(True)
        teacher.train(False)
        optimizer = lr_schedule.inv_lr_scheduler(optimizer,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, inputs_target_mosaic_w, inputs_target_mosaic_s, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs_target_mosaic_w, inputs_target_mosaic_s =  inputs_target_mosaic_w.cuda(), inputs_target_mosaic_s.cuda()
        # features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)
        with torch.no_grad():
            features_target_teacher, outputs_target_teacher = teacher(inputs_target)

        # classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        classifier_loss = 4*utils.cross_entropy_with_logits(outputs_target / 4.0, F.softmax(outputs_target_teacher / 4.0, dim=1))

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i)
        elif args.baseline =='DANN':
            lam = 0.0
        prob_max, pseu_labels_target = torch.max(F.softmax(outputs_target, dim=1), dim=1)

        total_loss = classifier_loss 
    
        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mosaic_loss_target = torch.zeros(1)

        with _disable_tracking_bn_stats(model):
            mosaic_features_target_w, mosaic_outputs_target_w = model(inputs_target_mosaic_w)
            mosaic_features_target_s, mosaic_outputs_target_s = model(inputs_target_mosaic_s)
            with torch.no_grad():
                
                features_list_w = [mosaic_features_target_w, F.softmax(mosaic_outputs_target_w, dim=1)]

                features_target_, outputs_target_ = model(inputs_target)
                prob_max, pseu_labels_target = torch.max(F.softmax(outputs_target_, dim=1), dim=1)
                features_list = [features_target_, F.softmax(outputs_target_, dim=1)]
                labels_list = [pseu_labels_target, pseu_labels_target]

                utils.rightshift(queue_weight, args.batch_size)
                for j in range(len(features_list)):
                    queue_data[j][queue_ptr:queue_ptr+args.batch_size, :] = features_list[j]
                    queue_data_w[j][queue_ptr:queue_ptr+args.batch_size, :] = features_list_w[j]
                    queue_labels[j][queue_ptr:queue_ptr+args.batch_size] = labels_list[j]
                pre_ptr = int(queue_ptr)
                ptr = ((i+1) % max_batch) * args.batch_size
                queue_ptr[0] = ptr

            if i < 10000:
                # mosaic_loss_target = nt_cent(queue_data[0].detach(), mosaic_features_target, queue_labels[0], 
                #     pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False)
                # mosaic_loss_target = (nt_cent(queue_data[0].detach(), mosaic_features_target_w, queue_labels[0], 
                #     pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False) +
                #                      1.*nt_cent(queue_data_w[0].detach(), mosaic_features_target_s, queue_labels[0], 
                #     pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False)) * (network.calc_coeff(max(i-max_batch, 0), high=1.0, max_iter=1000))

                mosaic_loss_target = (nt_cent(queue_data[1].detach(), F.softmax(mosaic_outputs_target_w, dim=1), queue_labels[1], 
                    pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False) +
                                     1.*nt_cent(queue_data_w[1].detach(), F.softmax(mosaic_outputs_target_s, dim=1), queue_labels[1], 
                    pseu_labels_target.float(), queue_weight, pre_ptr, class_level=False)) * network.calc_coeff(i, high=0.3, max_iter=1000)
            mosaic_loss = mosaic_loss_target * 1.0

            # mosaic_loss = utils.cross_entropy_with_logits(mosaic_outputs_target, F.softmax(outputs_target*1.5, dim=1)) * (network.calc_coeff(i, high=0.5, max_iter=2000))
            # mosaic_loss += 0.4*(torch.abs(F.softmax(outputs_target, dim=1).detach() - F.softmax(mosaic_outputs_target, dim=1)).sum(1)).mean(0)

            mosaic_loss.backward()
            optimizer.step()


        if i % print_interval == 0:
            log_str = 'step:{: d},\t,class_loss:{:.4f},\t,adv_loss:{:.4f}\t,mosaic_loss:{:.4f}\t,mean_prob:{:.4f}'.format(i, classifier_loss.item(),
                                                        0.0, mosaic_loss_target.item(),prob_max.mean().item())
            print(log_str)
            args.log_file.write('\n'+log_str)
            args.log_file.flush()

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model
