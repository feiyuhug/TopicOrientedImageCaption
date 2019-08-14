import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .ShowTellModel_t import ShowTellModel_t
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .Att2inModel import Att2inModel
from .AttModel import *
from .AttModel_t import *
from .AttModel_t_sc import *
from .AttModel_t_ens import *
from .FixAttModel2 import *

def setup(opt):

    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'show_tell_t':
        model = ShowTellModel_t(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        #assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.old_id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        if opt.start_from_best :
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
            print('model load from %s'%(os.path.join(opt.start_from, 'model-best.pth')))
        else :
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
            print('model load from %s'%(os.path.join(opt.start_from, 'model.pth')))

    return model
