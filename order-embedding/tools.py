"""
A selection of functions for encoding images and sentences
"""
import theano

import cPickle as pkl
import numpy

from collections import OrderedDict, defaultdict
from scipy.linalg import norm

from utils import load_params, init_tparams
from model import init_params, build_sentence_encoder, build_sentence_encoder_with_topicvector, build_image_encoder, build_topic_encoder, \
        build_topic_vector1_encoder, build_topic_vector2_encoder, build_errors, build_errors_3level, \
        build_errors_t2

def load_model(path_to_model):
    """
    Load all model components
    """
    print path_to_model

    # Load model
    print 'Loading model'
    with open(path_to_model + '.pkl', 'rb') as f:
        model = pkl.load(f)

    options = model['options']
    options['use_topic'] = True
    # Load parameters
    print 'Loading model parameters...'
    params = init_params(options)
    params = load_params(path_to_model + '.npz', params)
    tparams = init_tparams(params)

    # Extractor functions
    print 'Compiling sentence encoder...'
    [x, x_mask], sentences = build_sentence_encoder(tparams, options)
    f_senc = theano.function([x, x_mask], sentences, name='f_senc')

    #print 'Compiling sentence encoder with topics...'
    #[x, x_mask, topics], sentences = build_sentence_encoder_with_topicvector(tparams, options)
    #f_senc_t = theano.function([x, x_mask, topics], sentences, name='f_senc_t')

    print 'Compiling image encoder...'
    [im], images = build_image_encoder(tparams, options)
    f_ienc = theano.function([im], images, name='f_ienc')
    
    print 'Compiling topic encoder...'
    [t], topics = build_topic_encoder(tparams, options)
    f_tenc = theano.function([t], topics, name='f_tenc')

    '''
    print 'Compiling topic_vector1 encoder...'
    [t], topics = build_topic_vector1_encoder(tparams, options)
    f_tv1enc = theano.function([t], topics, name='f_tv1enc')

    print 'Compiling topic_vector2 encoder...'
    [t], topics = build_topic_encoder(tparams, options)
    f_tv2enc = theano.function([t], topics, name='f_tv2enc')
    '''
    print 'Compiling error computation...'
    [s, im], errs = build_errors(options)
    f_err = theano.function([s,im], errs, name='f_err')
    '''
    [s, im, t], errs_t1 = build_errors_3level(options)
    f_err_t1 = theano.function([s,im,t], errs_t1, name='f_err_t1')
    
    [s_t, im_t], errs_t2 = build_errors_t2(options)
    f_err_t2 = theano.function([s_t, im_t], errs_t2, name='f_err_t2')
    '''
    # Store everything we need in a dictionary
    print 'Packing up...'
    model['f_senc'] = f_senc
    #model['f_senc_t'] = f_senc_t
    model['f_ienc'] = f_ienc
    model['f_tenc'] = f_tenc
    #model['f_tv1enc'] = f_tv1enc
    #model['f_tv2enc'] = f_tv2enc
    model['f_err'] = f_err
    #model['f_err_t1'] = f_err_t1
    #model['f_err_t2'] = f_err_t2

    return model

def encode_sentences(model, X, verbose=False, batch_size=128):
    """
    Encode sentences into the joint embedding space
    """
    features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i, s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model['worddict'][w] if w in model['worddict'] and model['worddict'][w] < model['options']['n_words'] else 1 for w in cc])
            x = numpy.zeros((k+1, len(caption))).astype('int64')
            x_mask = numpy.zeros((k+1, len(caption))).astype('float32')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s
                x_mask[:k+1,idx] = 1.
            
            ff = model['f_senc'](x, x_mask)
            for ind, c in enumerate(caps):
                features[c] = ff[ind]

    return features

def encode_sentences_with_topicvector(model, X, tv1, verbose=False, batch_size=128):
    """
    Encode sentences into the joint embedding space
    """
    features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i, s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model['worddict'][w] if w in model['worddict'] and model['worddict'][w] < model['options']['n_words'] else 1 for w in cc])
            x = numpy.zeros((k+1, len(caption))).astype('int64')
            x_mask = numpy.zeros((k+1, len(caption))).astype('float32')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s
                x_mask[:k+1,idx] = 1.
            
            ff = model['f_senc_t'](x, x_mask, tv1[caps])
            for ind, c in enumerate(caps):
                features[c] = ff[ind]

    return features


def encode_images(model, IM):
    """
    Encode images into the joint embedding space
    """
    return model['f_ienc'](IM)

def encode_topics(model, TP):
    """
    Encode images into the joint embedding space
    """
    return model['f_tenc'](TP)

def encode_topic_vector1(model, tv1):
    return model['f_tv1enc'](tv1)

def encode_topic_vector2(model, tv2):
    return model['f_tv2enc'](tv2)

def compute_errors(model, s, im):
    """
    Computes errors between each sentence and caption
    """
    return model['f_err'](s, im)

def compute_errors_t1(model, s, im):
    return model['f_err_t1'](s, im)

def compute_errors_t2(model, s_t, im_t):
    return model['f_err_t2'](s_t, im_t)


