import os
import json
from copy import deepcopy
import numpy as np
import argparse
from keras.models import Model
from keras.utils import to_categorical
import keras
import tensorflow as tf

from utils import *
from optimizer import *
np.random.seed(0)


def attack(net_name, optimizer_name, start_id, end_id, gpu_id):
    # acquire basic information
    sess = tf.InteractiveSession()
    inputs, loss, grad = {}, {}, {}
    sample, prob, labels, result_dir, record, rmsds = {}, {}, {}, {}, [], []
    size, pre_pro = load_net(net_name, return_net=False)

    # hyper parameters
    # rmsd_thres = 7
    iter_thres = 10
    rmsd_thress = {
        'GD': 8,
        'MGD': 2,
        'NAGD': 2,
        'Adam': 4,
        'LAdam': 4,
        'MSVAG': 2,
        'AdaB': 4,
    }
    rmsd_first_thres = rmsd_thress[optimizer_name]
    epsilon = 16
    ao = Attack_Optimizer(optimizer_name)
    ############### alpha  = 2

    # record directory
    result_dir['base'] = f'{get_time()}_{net_name}_{optimizer_name}_CSTrmse_Iter{iter_thres}_RMSD{rmsd_first_thres}_Eps{epsilon}_Index{start_id}to{end_id}_GPU{gpu_id}'
    #for i in range(iter_thres+1): result_dir[i] = result_dir['base'] + '/adv/%d' % i
    result_dir['adv_final'] = result_dir['base']+ '/adv_final'
    result_dir['log'] = result_dir['base'] + '/log'
    for _, item in result_dir.items(): os.makedirs(item, exist_ok=True)
    copy_files(result_dir['base'] + '/src')
    log_file = open(result_dir['log'] + '/log.json', 'w')
    log = {}
    print('\n' + result_dir['base'], '\n')
    
    # build networks and LRP
    inputs['image']    = tf.placeholder(tf.float32, [1, size, size, 3], name='input')
    net, _             = load_net(net_name, inp=pre_pro(inputs['image'], backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils))
    prob_tensor        = net.output
    inputs['label']    = tf.placeholder(tf.float32, [1, 1000], name='label')
    
    # define loss
    loss['ce'] = - tf.reduce_mean(inputs['label'] * tf.log(prob_tensor + 1e-35))
    grad['ce'] = tf.gradients(loss['ce'], inputs['image'])[0]
    initialize_uninitialized(sess)
    
    # load label
    with open(paths['Label'], 'r') as f: label_list = f.read().split('\n')
    for item in label_list: 
        if item == '': continue
        item = item.split(' ')
        labels[item[0]] = int(item[1])

    # begin attack
    file_lists = sorted(os.listdir(paths['Data']))
    np.random.shuffle(file_lists)
    num_attack = end_id - start_id
    prt_detail = num_attack <= 100
    start = time.time()
    def save_imgs(img, path): PIL.Image.fromarray(img.astype(np.uint8)).save(path)
    losses = []
    for sam_id, file in enumerate(file_lists):
        # stop condition
        if sam_id < start_id: continue
        if sam_id >= end_id: break

        # load sample
        class_id = labels[file]
        sample['ori'] = process_sample(paths['Data'] + '/' + file, size)
        sample['adv'] = deepcopy(sample['ori'])
        
        # get original variables
        prob['pred']  = sess.run(prob_tensor, {inputs['image']: [sample['ori']]})[0]
        pred_ori = np.argmax(prob['pred'])
        prob['label'] = to_categorical(class_id, 1000)
        if np.argmax(prob['pred']) != class_id: 
            if prt_detail: print('Incorrect original Label (%d, %d)' % (np.argmax(prob['pred']), class_id))
            end_id += 1
            continue
            
        # attack iter for each sample
        success_condition = False
        error = False
        iter_done = 0
        loss_values, rmsd_values = [], []
        if prt_detail: print()
        while 1:
            # stop condition
            feed_dict = {inputs['image']: [sample['adv']], inputs['label']: [prob['label']]}
            rmsd = get_rmsd(sample)
            out_dict = {'Iter': iter_done, 'Success': success_condition, 'RMSD': rmsd}
            loss_value = sess.run(loss, feed_dict)
            out_dict.update(loss_value)
            loss_values.append(float(loss_value['ce']))
            rmsd_values.append(float(rmsd))
            output(out_dict, prt=prt_detail)
            #save_imgs(sample['adv'], result_dir[iter_done] + '/' + os.path.splitext(file)[0] + '.png')
            if iter_done >= iter_thres: break
            if np.isnan(np.mean(rmsd)): error = True; break

            # attack
            grad_func = lambda x: sess.run(grad['ce'], {inputs['image']: [x], inputs['label']: [prob['label']]})
            if iter_done == 0: 
                try: alpha = get_alpha(ao, rmsd_first_thres, grad_func, sample, epsilon) 
                except: error = True; break # Shampoo SVD error
            sample = ao.optimizer(grad_func, sample, iter_done, alpha, epsilon)
            prob['pred'] = sess.run(prob_tensor, {inputs['image']: [sample['adv']]})[0]
            
            # success condition
            success_condition = np.argmax(prob['pred']) != class_id
            iter_done += 1

        if error: 
            if prt_detail: print('encountered NAN')
            end_id += 1
            continue

        # visualization
        save_imgs(sample['adv'], result_dir['adv_final'] + '/' + os.path.splitext(file)[0] + '.png')
        record.append(success_condition)
        rmsds.append(rmsd)
        losses.append(loss_value['ce'])
        avg_rmsd = sum(rmsds)/(len(rmsds)+0.001)
        avg_loss = sum(losses)/(len(losses)+0.000001)
        output({'No':        '%d/%d' % (sam_id+1, end_id),
                'TimeRm':     convert_second_to_time((time.time()-start)/len(rmsds)*(num_attack-len(rmsds))),
                'RMSD':      '%.3f in %.3f' % (rmsd, avg_rmsd),
                'Loss':      '%.3f in %.3f' % (loss_value['ce'], avg_loss),
                'Lgafa':     np.log10(alpha),
                'Prob':      '%.2f->%.2f %.2f' % (np.max(prob['label']), prob['pred'][np.argmax(prob['label'])], np.max(prob['pred'])),
                'Class':     '%d %d->%d' % (np.argmax(prob['label']), pred_ori, np.argmax(prob['pred'])),
                })
        log[file] = {'alpha': float(alpha), 'loss': loss_values, 'rmsd': rmsd_values}
    json.dump(log, log_file)
    open(result_dir['log'] + '/Loss_%.3f_RMSE_%.3f.json' % (avg_loss, avg_rmsd), 'w')


def yield_task_parameters():
    for net_name in ['VGG16', 'ResNet50', 'InceptionV3', 'DenseNet121']:
        for optimizer_name in ['GD', 'MGD', 'NAGD', 'Adam', 'LAdam', 'MSVAG', 'AdaB', 'Shampoo']:
            yield net_name, optimizer_name, 0, 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('net_name', type=str, help='net_name')
    parser.add_argument('optimizer_name', type=str, help='optimizer_name')
    parser.add_argument('start_id', default=0, type=int, help='start_id')
    parser.add_argument('end_id', default=100, type=int, help='end_id')
    parser.add_argument('gpu_id', help='GPU(s) used')
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    attack(net_name=args.net_name, optimizer_name=args.optimizer_name, start_id=args.start_id, end_id=args.end_id, gpu_id=args.gpu_id)