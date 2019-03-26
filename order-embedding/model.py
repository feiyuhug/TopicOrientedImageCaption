"""
Model specification
"""
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal

from collections import OrderedDict

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, l2norm, maxnorm2
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()
    
    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Sentence encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])
    # topic_vector encoder1 to gru
    #params = get_layer('ff')[0](options, params, prefix='ff_topic_vector1_emb_gru', nin=options['dim_topic'], nout=options['dim'])

    # Image encoder
    params = get_layer('ff')[0](options, params, prefix='ff_image', nin=options['dim_image'], nout=options['dim'])
    
    # topic encoder
    if options['use_topic'] :
        params = get_layer('ff')[0](options, params, prefix='ff_topic', nin=options['dim_topic'], nout=options['dim'])
    
    '''   
    # topic_vector encoder1
    params = get_layer('ff')[0](options, params, prefix='ff_topic_vector1', nin=options['dim_topic'], nout=options['dim'])

    # topic_vector encoder2
    params = get_layer('ff')[0](options, params, prefix='ff_topic_vector2', nin=options['dim_topic'], nout=options['dim'])
    '''
    return params


def order_violations(s, im, options):
    """
    Computes the order violations (Equation 2 in the paper)
    """
    return tensor.pow(tensor.maximum(0, s - im), 2)


def contrastive_loss(s, im, options):
    """
    For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss
    """
    margin = options['margin']

    if options['method'] == 'order':
        im2 = im.dimshuffle(('x', 0, 1))
        s2 = s.dimshuffle((0, 'x', 1))
        errors = order_violations(s2, im2, options).sum(axis=2)
    elif options['method'] == 'cosine':
        errors = - tensor.dot(im, s.T) # negative because error is the opposite of (cosine) similarity

    diagonal = errors.diagonal()

    cost_s = tensor.maximum(0, margin - errors + diagonal)  # compare every diagonal score to scores in its column (all contrastive images for each sentence)
    cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))  # all contrastive sentences for each image

    cost_tot = cost_s + cost_im

    # clear diagonals
    cost_tot = fill_diagonal(cost_tot, 0)

    return cost_tot.sum()

def contrastive_loss_withmask(s, im, match_mask, options):
    """
    For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss
    """
    margin = options['margin']

    if options['method'] == 'order':
        im2 = im.dimshuffle(('x', 0, 1))
        s2 = s.dimshuffle((0, 'x', 1))
        errors = order_violations(s2, im2, options).sum(axis=2)
    elif options['method'] == 'cosine':
        errors = - tensor.dot(im, s.T) # negative because error is the opposite of (cosine) similarity

    diagonal = errors.diagonal()

    cost_s = tensor.maximum(0, margin - errors + diagonal)  # compare every diagonal score to scores in its column (all contrastive images for each sentence)
    cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))  # all contrastive sentences for each image

    cost_tot = cost_s + cost_im

    # clear diagonals
    #cost_tot = fill_diagonal(cost_tot, 0)
    cost_tot = cost_tot * match_mask

    return cost_tot.sum()


def encode_sentences(tparams, options, x, mask):
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    s = proj[0][-1]
    #if options['v_norm'] == 'l2' :
    s = l2norm(s)
    #s = maxnorm2(s)
    if options['abs']:
        #s = abs(s)
        s = tensor.maximum(s, 0)
    return s

def encode_sentences_with_topicvector(tparams, options, x, mask, topics):
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    t2gru_emb = get_layer('ff')[1](tparams, topics, options, prefix='ff_topic_vector1_emb_gru', activ='linear')
    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, t2gru_emb * 0.1, options,
                                            prefix='encoder',
                                            mask=mask)
    s = proj[0][-1]
    #if options['v_norm'] == 'l2' :
    s = l2norm(s)
    #s = maxnorm2(s)
    if options['abs']:
        #s = abs(s)
        s = tensor.maximum(s, 0)
    return s

def encode_images(tparams, options, im):
    im_emb = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')
    #if options['v_norm'] == 'l2' :
    im_emb = l2norm(im_emb)
    #im_emb = maxnorm2(im_emb)
    if options['abs']:
        #im_emb = abs(im_emb)
        im_emb = tensor.maximum(im_emb, 0)
        
    return im_emb

def encode_topics(tparams, options, topics):
    t_emb = get_layer('ff')[1](tparams, topics, options, prefix='ff_topic', activ='linear')
    #if options['v_norm'] == 'l2' :
    t_emb = l2norm(t_emb)
    #t_emb = maxnorm2(t_emb)
    if options['abs']:
        #im_emb = abs(im_emb)
        t_emb = tensor.maximum(t_emb, 0)
        
    return t_emb

def encode_topic_vector1(tparams, options, topics):
    t_emb = get_layer('ff')[1](tparams, topics, options, prefix='ff_topic_vector1', activ='linear')
    t_emb = l2norm(t_emb)
    #t_emb = maxnorm2(t_emb)

    if options['abs']:
        #im_emb = abs(im_emb)
        t_emb = tensor.maximum(t_emb, 0)
        
    return t_emb

def encode_topic_vector2(tparams, options, topics):
    t_emb = get_layer('ff')[1](tparams, topics, options, prefix='ff_topic_vector2', activ='linear')
    t_emb = l2norm(t_emb)
    #t_emb = maxnorm2(t_emb)

    if options['abs']:
        #im_emb = abs(im_emb)
        t_emb = tensor.maximum(t_emb, 0)
        
    return t_emb


def build_model(tparams, options):
    """
    Computation graph for the entire model
    """
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')
    
    #match_mask = tensor.matrix('match_mask', dtype='float32')
    # embed sentences and images
    s_emb = encode_sentences(tparams, options, x, mask)
    im_emb = encode_images(tparams, options, im)

    # Compute loss
    #@yuniange
    cost = contrastive_loss(s_emb, im_emb, options)
    #cost = contrastive_loss_withmask(s_emb, im_emb, match_mask, options)

    return [x, mask, im], cost

def build_model_reverse(tparams, options):
    """
    Computation graph for the entire model
    """
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')

    # embed sentences and images
    s_emb = encode_sentences(tparams, options, x, mask)
    im_emb = encode_images(tparams, options, im)

    # Compute loss
    #@yuniange
    cost = contrastive_loss(im_emb, s_emb, options)

    return [x, mask, im], cost

def build_model_3level(tparams, options):
    """
    Computation graph for the entire model
    """
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')
    topic = tensor.matrix('topic', dtype='float32')
    s_match_mask = tensor.matrix('s_match_mask', dtype='float32')
    im_match_mask = tensor.matrix('im_match_mask', dtype='float32')
    # embed sentences and images
    s_emb = encode_sentences(tparams, options, x, mask)
    im_emb = encode_images(tparams, options, im)
    t_emb = encode_topics(tparams, options, topic)

    # Compute loss
    #@yuniange
    cost = contrastive_loss(s_emb, im_emb, options) \
            + 0.2 * contrastive_loss_withmask(t_emb, im_emb, im_match_mask, options) \
            + 0.2 * contrastive_loss_withmask(t_emb, s_emb, s_match_mask, options)
    return [x, mask, im, topic, s_match_mask, im_match_mask], cost
    #return [x, mask, im, topic, im_match_mask], cost
    #return [x, mask, im, topic, s_match_mask], cost

def build_model_t2(tparams, options):
    """
    Computation graph for the entire model
    """
    s_t = tensor.matrix('s_t', dtype='float32')
    im_t = tensor.matrix('im_t', dtype='float32')

    # embed sentences and images
    s_t_emb = encode_topic_vector1(tparams, options, s_t)
    im_t_emb = encode_topic_vector2(tparams, options, im_t)
    # Compute loss
    #@yuniange
    cost = contrastive_loss(s_t_emb, im_t_emb, options)

    return [s_t, im_t], cost

def build_model_topic2gru(tparams, options):
    """
    Computation graph for the entire model
    """
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')
    s_t = tensor.matrix('s_t', dtype='float32')
    #match_mask = tensor.matrix('match_mask', dtype='float32')
    # embed sentences and images
    im_emb = encode_images(tparams, options, im)
    s_topic_emb = encode_sentences_with_topicvector(tparams, options, x, mask, s_t)
    # Compute loss
    #@yuniange
    cost = contrastive_loss(s_topic_emb, im_emb, options)
    #cost = contrastive_loss_withmask(s_emb, im_emb, match_mask, options)

    return [x, mask, im, s_t], cost


def build_sentence_encoder(tparams, options):
    """
    Encoder only, for sentences
    """
    # sentence features
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')

    return [x, mask], encode_sentences(tparams, options, x, mask)

def build_sentence_encoder_with_topicvector(tparams, options):
    """
    Encoder only, for sentences
    """
    # sentence features
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    topics = tensor.matrix('topics', dtype='float32')
    return [x, mask, topics], encode_sentences_with_topicvector(tparams, options, x, mask, topics)


def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    # image features
    im = tensor.matrix('im', dtype='float32')
    
    return [im], encode_images(tparams, options, im)

def build_topic_encoder(tparams, options):
    """
    Encoder only, for images
    """
    # image features
    t = tensor.matrix('t', dtype='float32')
    
    return [t], encode_topics(tparams, options, t)

def build_topic_vector1_encoder(tparams, options):
    """
    Encoder only, for images
    """
    # image features
    t = tensor.matrix('t', dtype='float32')
    
    return [t], encode_topic_vector1(tparams, options, t)

def build_topic_vector2_encoder(tparams, options):
    """
    Encoder only, for images
    """
    # image features
    t = tensor.matrix('t', dtype='float32')
    
    return [t], encode_topic_vector2(tparams, options, t)

def build_errors(options):
    """ Given sentence and image embeddings, compute the error matrix """
    # input features
    s_emb = tensor.matrix('s_emb', dtype='float32')
    im_emb = tensor.matrix('im_emb', dtype='float32')

    errs = None
    if options['method'] == 'order':
        # trick to make Theano not optimize this into a single matrix op, and overflow memory
        indices = tensor.arange(s_emb.shape[0])
        errs, _ = theano.map(lambda i, s, im: order_violations(s[i], im, options).sum(axis=1).flatten(),
                             sequences=[indices],
                             non_sequences=[s_emb, im_emb])
    else:
        errs = - tensor.dot(s_emb, im_emb.T)

    return [s_emb, im_emb], errs

def build_errors_3level(options):
    """ Given sentence and image embeddings, compute the error matrix """
    # input features
    s_emb = tensor.matrix('s_emb', dtype='float32')
    im_emb = tensor.matrix('im_emb', dtype='float32')
    #t_emb = tensor.matrix('t_emb', dtype='float32')

    errs = None
    if options['method'] == 'order':
        # trick to make Theano not optimize this into a single matrix op, and overflow memory
        indices = tensor.arange(s_emb.shape[0])
        errs, _ = theano.map(lambda i, s, im: order_violations(s[i], im, options).sum(axis=1).flatten(),
                             sequences=[indices],
                             non_sequences=[s_emb, im_emb])
        
    else:
        errs = - tensor.dot(s_emb, im_emb.T)

    return [s_emb, im_emb], errs

def build_errors_t2(options):
    # input features
    s_t_emb = tensor.matrix('s_t_emb', dtype='float32')
    im_t_emb = tensor.matrix('im_t_emb', dtype='float32')

    errs = None
    if options['method'] == 'order':
        # trick to make Theano not optimize this into a single matrix op, and overflow memory
        indices = tensor.arange(s_t_emb.shape[0])
        errs, _ = theano.map(lambda i, s, im: order_violations(s[i], im, options).sum(axis=1).flatten(),
                             sequences=[indices],
                             non_sequences=[s_t_emb, im_t_emb])
    else:
        errs = - tensor.dot(s_t_emb, im_t_emb.T)

    return [s_t_emb, im_t_emb], errs

