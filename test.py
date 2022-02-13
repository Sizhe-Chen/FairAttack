import os
import cv2
import json
import keras
import argparse
import PIL.Image
import numpy as np
import tensorflow as tf
import prettytable as pt
from copy import deepcopy
from scipy.stats import binom_test
from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader
from tensorflow.contrib.slim.nets import inception

from utils import *
from dfdmodels import nets, inception_resnet_v2

# wrap the network output
class Net:
    def __init__(self, output): self.output = output


# load feature-denoised network: cannot load both two models
def load_net_denoise(inp=None, model_id=0):
    def preprocess_input(x, backend, layers, models, utils):
        x = x / 127.5 - 1.0
        x = x[:, :, :, ::-1]
        x = tf.transpose(x, [0, 3, 1, 2])
        return x

    if inp is None: return 224, preprocess_input
    else:
        # if inp is specified, return the Net object net so that net.output is the [inp.shape[0], 1000] output tensor and prprocess function
        # P.S. the complicated design is to simplify API
        if model_id == 0:   model = nets.ResNeXtDenoiseAllModel()
        elif model_id == 1: model = nets.ResNetDenoiseModel()
        else: raise ValueError('Invalid Model ID', model_id)

        with TowerContext('', is_training=False): logits = model.get_logits(inp)
        sess = tf.get_default_session()

        if model_id == 0:   get_model_loader(paths['Model'] + '/X101-DenoiseAll.npz').init(sess)
        elif model_id == 1: get_model_loader(paths['Model'] + '/R152-Denoise.npz').init(sess)
        return Net(logits), preprocess_input


# for adv-trained models: cannot load two models with same model name
def create_model(x, model_name):
    if model_name == 'inception_v3':
        with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
            return inception.inception_v3(x, num_classes=1001, is_training=False)
    elif model_name == 'inception_resnet_v2':
        with tf.contrib.slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            return inception_resnet_v2.inception_resnet_v2(x, num_classes=1001, is_training=False)
    else: raise ValueError('Invalid model name: %s' % (model_name))


# load_inception_v3 adv-trained models
def load_inception_v3(inp=None, num_ens=0):
    from keras.applications.inception_v3 import preprocess_input
    if inp is None: return 299, preprocess_input
    old_variables = tf.global_variables()
    logits, _ = create_model(inp, 'inception_v3')
    sess = tf.get_default_session()
    if num_ens == 0:   model_path = paths['Model'] + '/adv_inception_v3.ckpt'
    elif num_ens == 3: model_path = paths['Model'] + '/ens3_adv_inception_v3.ckpt'
    elif num_ens == 4: model_path = paths['Model'] + '/ens4_adv_inception_v3.ckpt'
    else: raise ValueError('Invalid ensemble number: %d' % num_ens)
    tf.train.Saver([x for x in tf.global_variables() if x not in old_variables]).restore(sess, model_path)
    return Net(logits), preprocess_input


# load_inception_resnet_v2 adv-trained models
def load_inception_resnet_v2(inp=None, num_ens=0):
    from keras_applications.inception_resnet_v2 import preprocess_input
    if inp is None: return 299, preprocess_input
    old_variables = tf.global_variables()
    logits, _ = create_model(inp, 'inception_resnet_v2')
    sess = tf.get_default_session()
    if num_ens == 0:   model_path = paths['Model'] + '/adv_inception_resnet_v2.ckpt'
    elif num_ens == 1: model_path = paths['Model'] + '/ens_adv_inception_resnet_v2.ckpt'
    else: raise ValueError('Invalid ensemble number: %d' % num_ens)
    tf.train.Saver([x for x in tf.global_variables() if x not in old_variables]).restore(sess, model_path)
    return Net(logits), preprocess_input


# load the network
def load_net_info(net_name, inp=None):
    size = {'InceptionV3': 299, 'Xception': 299, 'NASNetLarge': 331}.get(net_name, 224) # corresponding input size
    if 'adv' in net_name: size = 299
    if inp is not None and int(inp.shape[2]) != size:
        print('The input is resized from', inp.shape[1], 'to', size, ', which may leads to great transferability drop.')
        inp = tf.image.resize_bilinear(inp, (size, size))
    if   net_name == 'ResNet50':        from keras.applications.resnet50            import ResNet50, preprocess_input;          net = ResNet50(input_tensor=inp)    if inp is not None else size
    elif net_name == 'ResNet101':       from keras_applications.resnet_v2           import ResNet101V2, preprocess_input;       net = ResNet101V2(input_tensor=inp, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)       if inp is not None else size
    elif net_name == 'ResNet152':       from keras_applications.resnet_v2           import ResNet152V2, preprocess_input;       net = ResNet152V2(input_tensor=inp, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)       if inp is not None else size
    elif net_name == 'InceptionResNetV2': from keras_applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input; net = InceptionResNetV2(input_tensor=inp, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils) if inp is not None else size
    elif net_name == 'InceptionV3':     from keras.applications.inception_v3        import InceptionV3, preprocess_input;       net = InceptionV3(input_tensor=inp) if inp is not None else size
    elif net_name == 'Xception':        from keras.applications.xception            import Xception, preprocess_input;          net = Xception(input_tensor=inp)    if inp is not None else size
    elif net_name == 'VGG16':           from keras.applications.vgg16               import VGG16, preprocess_input;             net = VGG16(input_tensor=inp)       if inp is not None else size
    elif net_name == 'VGG19':           from keras.applications.vgg19               import VGG19, preprocess_input;             net = VGG19(input_tensor=inp)       if inp is not None else size
    elif net_name == 'DenseNet121':     from keras.applications.densenet            import DenseNet121, preprocess_input;       net = DenseNet121(input_tensor=inp) if inp is not None else size
    elif net_name == 'DenseNet169':     from keras.applications.densenet            import DenseNet169, preprocess_input;       net = DenseNet169(input_tensor=inp) if inp is not None else size
    elif net_name == 'DenseNet201':     from keras.applications.densenet            import DenseNet201, preprocess_input;       net = DenseNet201(input_tensor=inp) if inp is not None else size
    elif net_name == 'NASNetMobile':    from keras.applications.nasnet              import NASNetMobile, preprocess_input;      net = NASNetMobile(input_tensor=inp)if inp is not None else size
    elif net_name == 'NASNetLarge':     from keras.applications.nasnet              import NASNetLarge, preprocess_input;       net = NASNetLarge(input_tensor=inp) if inp is not None else size
    elif net_name == 'InceptionV3adv':          net, preprocess_input = load_inception_v3(inp=inp, num_ens=0)
    elif net_name == 'InceptionV3advens3':      net, preprocess_input = load_inception_v3(inp=inp, num_ens=3)
    elif net_name == 'InceptionV3advens4':      net, preprocess_input = load_inception_v3(inp=inp, num_ens=4)
    elif net_name == 'InceptionResNetV2adv':    net, preprocess_input = load_inception_resnet_v2(inp=inp, num_ens=0)
    elif net_name == 'InceptionResNetV2advens': net, preprocess_input = load_inception_resnet_v2(inp=inp, num_ens=1)
    elif net_name == 'ResNetXt101denoise':      net, preprocess_input = load_net_denoise(inp=inp, model_id=0)
    elif net_name == 'ResNet152denoise':        net, preprocess_input = load_net_denoise(inp=inp, model_id=1)
    else: raise ValueError('Invalid Network Name')
    return net, preprocess_input


# build all networks
def build(net_list):
    sess = tf.InteractiveSession()
    inputs, outputs, size = {}, {}, {} # record variables in dict
    for n in net_list:
        print('Loading', n)
        size[n], pre_pro = load_net_info(n) # load size and preprocessing function
        inputs[n]  = tf.placeholder(tf.float32, [1, size[n], size[n], 3], name=n)
        net, _  = load_net_info(n, pre_pro(inputs[n], backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils))
        outputs[n] = net.output
    return sess, inputs, outputs, size


# main function
def test(dataset, net_list, sess, inputs, outputs, size):
    # load labels
    labels = {}
    with open(paths['Label'], 'r') as f: label_list = f.read().replace('JPEG', 'png').split('\n')
    for item in label_list: 
        if item == '': continue
        item = item.split(' ')
        labels[item[0]] = int(item[1])
    
    # pred with or without random_smoothing
    def get_prediction(image, net_name, do_rs):
        def pred(img): return np.argmax(sess.run(outputs[net_name], {inputs[net_name]: [crop_or_pad(img, size[net_name])]})[0])
        if not do_rs: return pred(image)
        else:
            preds = []
            for i in range(100):
                preds.append(pred(np.clip(image + np.random.normal(scale=0.25 * 255, size=image.shape), 0, 255)))
            cnts = dict((a, preds.count(a)) for a in preds)
            na = [(k, v) for k, v in cnts.items() if sorted(cnts.values())[-1] == v][0]
            try: nb = [(k, v) for k, v in cnts.items() if sorted(cnts.values())[-2] == v][0]
            except IndexError: nb = (0, 0)
            if binom_test(na[1], na[1] + nb[1], 0.5) <= 0.001: return na[0]
            else: return None

    # if other defense exists, test random smoothing
    result_dirname = 'result'
    dirs = ['adv/%d' % i for i in range(11)] #['adv/10', 'defense/jpeg', 'defense/pixel', 'defense/random', 'defense/tvm', 'defense/smoothing'] ##############################
    if os.path.exists(dataset + '/adv_final'): dirs = ['adv_final']
    is_targeted = 'T1' in dataset
    random_smoothing = False#os.path.exists(dataset + '/' + dirs[-2])
    if random_smoothing: os.makedirs(dataset + '/' + dirs[-1], exist_ok=True)
    for directory in reversed(dirs):
        source_path = dataset + '/' + directory
        if not os.path.exists(source_path): continue
        print(source_path)
        os.makedirs(dataset + '/' + result_dirname, exist_ok=True)
        log = {}
        for net in net_list: log[net] = []
        file_list = os.listdir(source_path)
        do_random_smoothing = file_list == [] and random_smoothing
        if do_random_smoothing:
            source_path = dataset + '/' + dirs[0]
            file_list = os.listdir(source_path)
        output_file = open(dataset + '/' + result_dirname + '/final_result_%s_%d.txt' % (directory.replace('/', '_'), len(file_list)), 'w')
        
        # pred for each sample in each network
        for i, file in enumerate(file_list):
            class_id = labels[file.replace('jpg', 'png')]
            for n in net_list:
                sample = process_sample(source_path + '/' + file, size[n])
                prd = get_prediction(sample, n, do_random_smoothing)
                if prd is not None:
                    if is_targeted: do_error = prd == (class_id + 500 + int(outputs[n].shape[1]) - 1000) % 1000
                    else:           do_error = (prd != class_id + int(outputs[n].shape[1]) - 1000) and (prd != int(outputs[n].shape[1]) - 1001)
                    log[n].append(do_error)
            err_str = ''
            for net in log: err_str += '%.2f' % (sum(log[net])/(len(log[net])+0.01)*100) + '% '
            out = {'Sample': '%d/%d' % (i+1, len(file_list)), 'Error': err_str}
            output(out, end='\r')
        
        # save results
        tb = pt.PrettyTable()
        tb.field_names = ["Network", "Top-1 Error"]
        log_out = {}
        for net in net_list:
            length = len(log[net])
            batch = int(length/5)
            ratess = []
            for split in range(5):
                ratess.append(sum(log[net][split * batch: (split + 1) * batch]) / batch * 100)
            result = '%.1f+-%.1f' % (sum(log[net]) / length * 100, np.std(ratess)) + '%'
            tb.add_row([net, result])
            log_out[net] = result
        print(' ' * len(err_str), end='\r')
        print(tb)
        print(tb, file=output_file)
        output_file.close()
        json.dump(log_out, open(dataset + '/%s/final_result_%s_%d.json' % (result_dirname, directory.replace('/', '_'), len(file_list)), 'w'))


# test all directorys starting from start_id
def test_all(start_id, net_list):
    datasets = [x for x in os.listdir('.') if '202' in x and 'log' not in x]
    datasets.sort(key=lambda x: time.strptime(x[:19], '%Y-%m-%d-%H-%M-%S'))
    datasets = datasets[start_id:]
    print('Test', net_list, 'in', datasets)
    sess, inputs, outputs, size = build(net_list)
    for dataset in datasets: test(dataset, net_list, sess, inputs, outputs, size)


# automatically test
def auto_test(net_list):
    datasets = [x for x in os.listdir('.') if '202' in x and 'log' not in x]
    datasets.sort(key=lambda x: time.strptime(x[:19], '%Y-%m-%d-%H-%M-%S'))
    sess, inputs, outputs, size = build(net_list)
    for dataset in datasets:
        dir_names = os.listdir(dataset)
        if 'final_result_adv_1000.txt' in dir_names:
            if 'defense' not in dir_names: do_continue = True
            else:
                do_continue = True
                for dir_defense in os.listdir(dataset + '/defense'):
                    if 'final_result_defense_%s_1000.txt' % dir_defense not in dir_names: do_continue = False
            if do_continue: print('Skip tested dir', dataset); continue
        #if len(os.listdir(dataset + '/adv')) < 1000: print('Only', len(os.listdir(dataset + '/adv')), '< 1000 files in', dataset)
        test(dataset, net_list, sess, inputs, outputs, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='int -> test all starting from this, auto -> auto test, dir_name -> test this directory')
    parser.add_argument('gpu_id', help='GPU(s) used')
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    net_list = ['DenseNet201', 'VGG19', 'ResNet152', 'InceptionResNetV2', 'Xception', 'NASNetLarge', 'InceptionV3adv', 'InceptionResNetV2adv', 'ResNetXt101denoise']
    try: test_all(start_id=int(args.dataset), net_list=net_list)
    except ValueError: 
        if args.dataset == 'auto': auto_test(net_list)
        elif args.dataset == '': exit()
        else:
            sess, inputs, outputs, size = build(net_list)
            for dataset in args.dataset.split(','):
                test(dataset, net_list, sess, inputs, outputs, size)