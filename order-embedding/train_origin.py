"""
Main trainer function
"""
import theano

import cPickle as pkl
import json

import os
import time

import datasource

from utils import *
from optim import adam
from model import init_params, build_model, build_sentence_encoder, build_image_encoder, build_errors
from vocab import build_dictionary
from evaluation import t2i, i2t
from tools import encode_sentences, encode_images, compute_errors
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

    # initialize logger
    import datetime
    timestampedName = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + name

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
    dev_caps, dev_ims = dev.all()

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if load_from is not None and os.path.exists(load_from):
        params = load_params(load_from, params)

    tparams = init_tparams(params)

    inps, cost = build_model(tparams, model_options)

    print 'Building sentence encoder'
    inps_se, sentences = build_sentence_encoder(tparams, model_options)
    f_senc = theano.function(inps_se, sentences, profile=False)

    print 'Building image encoder'
    inps_ie, images = build_image_encoder(tparams, model_options)
    f_ienc = theano.function(inps_ie, images, profile=False)

    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))

    print 'Building errors..'
    inps_err, errs = build_errors(model_options)
    f_err = theano.function(inps_err, errs, profile=False)

    curr_model['f_senc'] = f_senc
    curr_model['f_ienc'] = f_ienc
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


    
    for eidx in xrange(model_options['max_epochs']):

        print 'Epoch ', eidx

        for x, mask, im in train_iter:
            n_samples += x.shape[1]
            uidx += 1

            # Update
            ud_start = time.time()
            cost = f_grad_shared(x, mask, im)
            f_update(model_options['lrate'])
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, model_options['dispFreq']) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                log.update({'Error': float(cost)}, n_samples)


            if numpy.mod(uidx, model_options['validFreq']) == 0:

                print 'Computing results...'

                # encode sentences efficiently
                dev_s = encode_sentences(curr_model, dev_caps, batch_size=model_options['batch_size'])
                dev_i = encode_images(curr_model, dev_ims)


                # compute errors
                dev_errs = compute_errors(curr_model, dev_s, dev_i)

                # compute ranking error
                (r1, r5, r10, medr, meanr), vis_details = t2i(dev_errs, vis_details=True)
                (r1i, r5i, r10i, medri, meanri) = i2t(dev_errs)
                print "Text to image (dev set): %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr)
                log.update({'R@1': r1, 'R@5': r5, 'R@10': r10, 'median_rank': medr, 'mean_rank': meanr}, n_samples)
                print "Image to text (dev set): %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri)
                log.update({'Image2Caption_R@1': r1i, 'Image2Caption_R@5': r5i, 'Image2CaptionR@10': r10i, 'Image2Caption_median_rank': medri, 'Image2Caption_mean_rank': meanri}, n_samples)

                tot = r1 + r5 + r10
                if tot > curr:
                    curr = tot
                    # Save parameters
                    print 'Saving...',
                    numpy.savez('%s/%s'%(save_dir, name), **unzip(tparams))
                    print 'Done'
                    vis_details['hyperparams'] = model_options
                    # Save visualization details
                    with open('vis/roc/%s/%s.json' % (model_options['data'], timestampedName), 'w') as f:
                        json.dump(vis_details, f)
                    # Add the new model to the index
		    try: 
                    	index = json.load(open('vis/roc/index.json', 'r'))
		    except IOError:
			index = { model_options['data'] : [] }

                    models = index[model_options['data']]
                    if timestampedName not in models:
                        models.append(timestampedName)

                    with open('vis/roc/index.json', 'w') as f:
                        json.dump(index, f)






        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass

