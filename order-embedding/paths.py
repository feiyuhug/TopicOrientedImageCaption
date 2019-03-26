import numpy

# location for captions and image CNN features
dataset_dir = {
    'coco': 'data/coco',
    'flickr30k': 'data/flickr30k'
}


# Change paths below only if you are computing your own image CNN features

images_dir = {
    'coco': '/ais/gobi3/datasets/mscoco/images',  # location for raw images
    'flickr30k': '../data/flickr30k/images'
}

cnns = {
    'VGG19':
            {
                'prototxt': '../multi-label/model_vgg19/model_vgg19_deploy.prototxt',
                'caffemodel': '../multi-label/VGG_ILSVRC_19_layers.caffemodel',
                'features_layer': 'fc7',
                'mean': numpy.array([103.939, 116.779, 123.68])  # BGR means, from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
            }
}
