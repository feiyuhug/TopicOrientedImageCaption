"""
Main trainer function
"""
import theano
import numpy as np
import random

import cPickle as pkl
import json

import os
import time

import datasource

from utils import *
from optim import adam
from model import init_params, build_model, build_model_reverse, build_model_3level, build_model_t2, build_model_topic2gru, build_sentence_encoder, build_sentence_encoder_with_topicvector, build_image_encoder, build_topic_encoder, build_topic_vector1_encoder, build_topic_vector2_encoder, build_errors, build_errors_3level, build_errors_t2
from vocab import build_dictionary
from evaluation import t2i, i2t
from tools import encode_sentences, encode_sentences_with_topicvector, encode_images, encode_topics, encode_topic_vector1, encode_topic_vector2, compute_errors, compute_errors_t1, compute_errors_t2
from datasets import load_dataset

# main trainer
def trainer(load_from=None,
            save_dir='snapshots',
            name='anon',
            **kwargs):
    """
    :param load_from: location to load parameters + options from
    :param name: name of model, used as location to save parameters + options
    """

    """
    kwargs:
    load_from   %save_dir/name
    data    coco
    cnn 10crop
    batch_size  *num

    """
    debug = True
    order_reverse = False
    curr_model = dict()

    # load old model, including parameters, but overwrite with new options
    if load_from:
        print 'reloading...' + load_from
        with open('%s.pkl'%load_from, 'rb') as f:
            curr_model = pkl.load(f)
    else:
        curr_model['options'] = {}

    for k, v in kwargs.iteritems():
        curr_model['options'][k] = v

    model_options = curr_model['options']
    topic_insert = 'none'
    if model_options['use_topic'] :
        topic_insert = 't1'

    # initialize logger
    
    import datetime
    timestampedName = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + name
    if debug :
        from logger import Log
        log = Log(name=timestampedName, hyperparams=model_options, saveDir='vis/training',
                  xLabel='Examples Seen', saveFrequency=1)


    print curr_model['options']




    # Load training and development sets
    print 'Loading dataset'
    dataset = load_dataset(model_options['data'], cnn=model_options['cnn'], load_train=True)
    train = dataset['train']
    dev = dataset['dev']    #val set

    # Create dictionary
    print 'Creating dictionary'
    worddict = build_dictionary(train['caps']+dev['caps'])  # 0 for <eos>, 1 for <unk>, sorted by frequency goes down

    print 'Dictionary size: ' + str(len(worddict))
    curr_model['worddict'] = worddict
    curr_model['options']['n_words'] = len(worddict) + 2

    # save model
    pkl.dump(curr_model, open('%s/%s.pkl' % (save_dir, name), 'wb'))


    print 'Loading data'
    train_iter = datasource.Datasource(train, batch_size=model_options['batch_size'], worddict=worddict)
    dev = datasource.Datasource(dev, worddict=worddict)
    dev_caps, dev_ims, dev_cap_tps, dev_im_tps = dev.all()

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if load_from is not None:
        if os.path.exists('%s.npz'%(load_from)):
            params = load_params('%s.npz'%(load_from), params)
            print 'reload params successful!'
        else :
            print '%s.npz not exists'%(load_from)

    tparams = init_tparams(params)
    
    if topic_insert == 't1' :
        inps, cost = build_model_3level(tparams, model_options)
    elif topic_insert == 't2' :
        inps, cost = build_model_t2(tparams, model_options)
    elif topic_insert == 't3' :
        inps, cost = build_model_topic2gru(tparams, model_options)
    elif order_reverse :
        inps, cost = build_model_reverse(tparams, model_options)
    else :
        inps, cost = build_model(tparams, model_options)
     
    print 'Building sentence encoder'
    inps_se, sentences = build_sentence_encoder(tparams, model_options)
    f_senc = theano.function(inps_se, sentences, profile=False)
    
    #print 'Compiling sentence encoder with topics...'
    #[x, x_mask, topics], sentences = build_sentence_encoder_with_topicvector(tparams, model_options)
    #f_senc_t = theano.function([x, x_mask, topics], sentences, name='f_senc_t')

    print 'Building image encoder'
    inps_ie, images = build_image_encoder(tparams, model_options)
    f_ienc = theano.function(inps_ie, images, profile=False)
    if model_options['use_topic'] :
        print 'Building topic encoder'
        inps_te, topics = build_topic_encoder(tparams, model_options)
        f_tenc = theano.function(inps_te, topics, profile=False)
    
    '''
    print 'Building topic vector1 encoder'
    inps_tv1e, topic_vector1 = build_topic_vector1_encoder(tparams, model_options)
    f_tv1enc = theano.function(inps_tv1e, topic_vector1, profile=False)

    print 'Building topic vector2 encoder'
    inps_tv2e, topic_vector2 = build_topic_vector2_encoder(tparams, model_options)
    f_tv2enc = theano.function(inps_tv2e, topic_vector2, profile=False)
    '''
    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    
    curr_model['f_senc'] = f_senc
    #curr_model['f_senc_t'] = f_senc_t
    curr_model['f_ienc'] = f_ienc
    if model_options['use_topic'] :
        curr_model['f_tenc'] = f_tenc
    '''
    curr_model['f_tv1enc'] = f_tv1enc
    curr_model['f_tv2enc'] = f_tv2enc
    '''
    print 'Building errors..'
    if topic_insert == 't1' :
        inps_err, errs_t1 = build_errors_3level(model_options)
        f_err_t1 = theano.function(inps_err, errs_t1, profile=False)
        curr_model['f_err_t1'] = f_err_t1
    elif topic_insert == 't2' :
        inps_err, errs_t2 = build_errors_t2(model_options)
        f_err_t2 = theano.function(inps_err, errs_t2, profile=False)
        curr_model['f_err_t2'] = f_err_t2
    else :
        inps_err, errs = build_errors(model_options)
        f_err = theano.function(inps_err, errs, profile=False)
        curr_model['f_err'] = f_err

    if model_options['grad_clip'] > 0.:
        grads = [maxnorm(g, model_options['grad_clip']) for g in grads]

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(model_options['optimizer'])(lr, tparams, grads, inps, cost)

    print 'Optimization'

    uidx = 0
    curr = 0
    n_samples = 0

    topic_encode = np.eye(model_options['dim_topic'],dtype='float32')

    for eidx in xrange(model_options['max_epochs']):

        print 'Epoch ', eidx

        for im_ind, x, mask, im, cap_tp, im_tp in train_iter:  #!!!!mask only in training stage, not in val and test
            n_samples += x.shape[1]
            uidx += 1
            # norm topic confidence
            if model_options['use_topic'] :
                s_match_mask_ = (cap_tp < 0.1).astype('float32').T
                im_match_mask_ = (im_tp < 0.1).astype('float32').T
                
                topic_idx = []
                for i in range(cap_tp.shape[0]) :
                    candidate_topics = [ind for (ind, conf) in zip(range(cap_tp.shape[1]), cap_tp[i].tolist()) if conf > 0.3]
                    if(len(candidate_topics) == 0) :
                        s_match_mask_[:,i] = 0
                        im_match_mask_[:,i] = 0
                        candidate_topic = random.randrange(model_options['dim_topic'])
                        s_match_mask_[candidate_topic, :] = 0
                        im_match_mask_[candidate_topic, :] = 0
                    else :
                        candidate_topic = candidate_topics[random.randrange(len(candidate_topics))] 
                    topic_idx.append(candidate_topic)
                
                topic_idx = np.array(topic_idx, dtype='int32')
                topic_input = topic_encode[topic_idx]
                s_match_mask = s_match_mask_[topic_idx]
                im_match_mask = im_match_mask_[topic_idx]
            # Update
            ud_start = time.time()
            if topic_insert == 't1' :
                cost = f_grad_shared(x, mask, im, topic_input, s_match_mask, im_match_mask)
                #cost = f_grad_shared(x, mask, im, topic_input, s_match_mask)
                #cost = f_grad_shared(x, mask, im, topic_input, im_match_mask)
            elif topic_insert == 't2' :
                cost = f_grad_shared(cap_tp, im_tp)
            elif topic_insert == 't3' :
                cost = f_grad_shared(x, mask, im, cap_tp)
            else :
                cost = f_grad_shared(x, mask, im)
            f_update(model_options['lrate'])
            ud = time.time() - ud_start
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                print 'Seen %d samples'%n_samples
                return 1., 1., 1.

            if numpy.mod(uidx, model_options['dispFreq']) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                if debug :
                    log.update({'Error': float(cost)}, n_samples)


            if numpy.mod(uidx, model_options['validFreq']) == 0:

                print 'Computing results...'

                # encode sentences efficiently
                dev_s = encode_sentences(curr_model, dev_caps, batch_size=model_options['batch_size'])
                #dev_s = encode_sentences_with_topicvector(curr_model, dev_caps, dev_cap_tps, batch_size=model_options['batch_size'])
                dev_i = encode_images(curr_model, dev_ims)
                #dev_ct = encode_topics(curr_model, dev_cap_tps)
                #dev_st = encode_topic_vector1(curr_model, dev_cap_tps)
                #dev_imt = encode_topic_vector2(curr_model, dev_im_tps)
                # compute errors
                if topic_insert == 't1' :
                    dev_errs = compute_errors_t1(curr_model, dev_s, dev_i)
                elif topic_insert == 't2' :
                    dev_errs = compute_errors_t2(curr_model, dev_st, dev_imt)
                elif order_reverse :
                    dev_errs = compute_errors(curr_model, dev_i, dev_s)
                    dev_errs = dev_errs.T
                else :
                    dev_errs = compute_errors(curr_model, dev_s, dev_i)
                # compute ranking error
                (r1, r5, r10, medr, meanr), vis_details = t2i(dev_errs, vis_details=True)
                (r1i, r5i, r10i, medri, meanri) = i2t(dev_errs)
                print "Text to image (dev set): %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr)
                if debug :
                    log.update({'R@1': r1, 'R@5': r5, 'R@10': r10, 'median_rank': medr, 'mean_rank': meanr}, n_samples)
                print "Image to text (dev set): %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri)
                if debug :
                    log.update({'Image2Caption_R@1': r1i, 'Image2Caption_R@5': r5i, 'Image2CaptionR@10': r10i, 'Image2Caption_median_rank': medri, 'Image2Caption_mean_rank': meanri}, n_samples)

                tot = r1 + r5 + r10
                if eidx % 5 == 0:
                    numpy.savez('%s/%s_%d'%(save_dir, name, eidx), **unzip(tparams))
                if tot > curr and debug:
                    curr = tot
                    # Save parameters
                    print 'Saving...',
                    numpy.savez('%s/%s'%(save_dir, name), **unzip(tparams))
                    print 'Done'
                    
                    vis_details['hyperparams'] = model_options
                    # Save visualization details
                    with open('vis/roc/%s/%s.json' % (model_options['data'], timestampedName), 'w') as f:
                        json.dump(vis_details, f)

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass

