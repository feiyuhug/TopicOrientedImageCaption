
require 'torch'
require 'nn'
require 'nngraph'
require 'npy4th'
-- exotic things
require 'loadcaffe'
--caffegraph = require 'caffegraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoader_test2014'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','coco/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_topics','coco/data_topics.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','coco/data.json','path to the json file containing additional info and vocab')
cmd:option('-input_h52','coco/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_topics2','coco/data_topics.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json2','coco/data.json','path to the json file containing additional info and vocab')

cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-fcn_proto','model/topic_emb.prototxt','path to fcn prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-fcn_model','model/topic_emb_0.2.caffemodel','path to fcn model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-topic_num', 100, 'number of topics')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',50,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-finetune_cnn_fcn_w_after', -1, '')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
--cmd:option('-cnn_learning_rate',2e-6,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
-- Optimization: for the fcn
cmd:option('-fcn_optim','adam','optimization to use for CNN')
cmd:option('-fcn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-fcn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-fcn_learning_rate',1e-4,'learning rate for the CNN')
cmd:option('-fcn_weight_decay', 0, 'L2 weight decay just for the CNN')
-- Optimization: for cnn_w and fcn_w
cmd:option('-cnn_fcn_w_optim','adam','optimization to use for CNN')
cmd:option('-cnn_fcn_w_optim_alpha', 0.8, '')
cmd:option('-cnn_fcn_w_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_fcn_w_learning_rate',4e-3,'learning rate for the CNN')


-- Evaluation/Checkpointing
cmd:option('-val_images_use', 5000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

-- new added
cmd:option('-beam_size', 3, 'beam size for test')
cmd:option('-net_type', 2, '1 for vgg, 2 for resnet')
cmd:option('-eval_test2014', 0, '0 for not use, 1 for use')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, h5_file_topics=opt.input_h5_topics, json_file = opt.input_json, topic_num = opt.topic_num}
if opt.eval_test2014 > 0 then
  loader2 = DataLoader_test2014{h5_file = opt.input_h52, h5_file_topics=opt.input_h5_topics2, json_file = opt.input_json2, topic_num = opt.topic_num}
end
print('dataloader done...')
-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  net_utils.unsanitize_gradients(protos.fcn)
  net_utils.unsanitize_gradients(protos.cnn_w)
  net_utils.unsanitize_gradients(protos.fcn_w)
  net_utils.unsanitize_gradients(protos.mapper)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
  protos.combiner = nn.CAddTable()
else
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  protos.lm = nn.LanguageModel(lmOpt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  
  -- initialize the topic fcn
  if opt.net_type == 1 then
    local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
    protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend, layer_num = 45})
    local fcn_raw = loadcaffe.load(opt.fcn_proto, opt.fcn_model, cnn_backend)
    protos.fcn = net_utils.build_fcn(fcn_raw, {backend = cnn_backend, layer_num = 2})
  else 
    local cnn_raw = caffegraph.load(opt.cnn_proto, opt.cnn_model)
    protos.cnn = net_utils.build_cnn_resnet(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend, layer_num = 45})
    net_utils.forth_unsanitize_gradients(cnn_raw)
    protos.fcn = net_utils.build_fcn_fromt7()
  end
  print(protos.cnn)
  print(protos.fcn)
  -- combine cnn and topic fcn
  protos.cnn_w = nn.CMul(1, 1024)
  protos.cnn_w.weight = (torch.ones(protos.cnn_w.weight:size()) * 1.0):float()
  protos.fcn_w = nn.CMul(1, 1024)
  protos.fcn_w.weight = (torch.ones(protos.fcn_w.weight:size()) * 0.2):float()

  protos.combiner = nn.CAddTable()
  protos.mapper = nn.Linear(1024, opt.input_encoding_size)
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
local fcn_params, fcn_grad_params = protos.fcn:getParameters()
local cnn_w_params, cnn_w_grad_params = protos.cnn_w:getParameters()
local fcn_w_params, fcn_w_grad_params = protos.fcn_w:getParameters()
local map_params, map_grad_params = protos.mapper:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
print('total number of parameters in fcn: ', fcn_params:nElement())
print('total number of parameters in cnn_w: ', cnn_w_params:nElement())
print('total number of parameters in fcn_w: ', fcn_w_params:nElement())
print('total number of parameters in map: ', map_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())
assert(fcn_params:nElement() == fcn_grad_params:nElement())
assert(map_params:nElement() == map_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
if opt.net_type == 2 then 
  cnn_module_list = net_utils.list_nngraph_modules(protos.cnn)
  raw_cnn_module_list = net_utils.list_nngraph_modules(cnn_module_list[2])
  thin_cnn_module_list = net_utils.list_nngraph_modules(thin_cnn)
  thin_raw_cnn_module_list = net_utils.list_nngraph_modules(thin_cnn_module_list[2])
  for k,m in ipairs(raw_cnn_module_list) do
    if m.running_mean then
      print(m, m.running_mean:mean(), m.running_var:mean())
      thin_raw_cnn_module_list[k].running_mean = m.running_mean
      thin_raw_cnn_module_list[k].running_var = m.running_var
    end
  end
end

local thin_fcn = protos.fcn:clone('weight', 'bias')
local thin_cnn_w = protos.cnn_w:clone('weight', 'bias')
local thin_fcn_w = protos.fcn_w:clone('weight', 'bias')
local thin_map = protos.mapper:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
net_utils.sanitize_gradients(thin_fcn)
net_utils.sanitize_gradients(thin_map)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end
npy4th = require 'npy4th'
net_utils.resnet_mean = npy4th.loadnpy('resnet_mean_RGB.npy'):cuda()
-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  if opt.net_type == 1 then
    protos.cnn:evaluate()
  else 
    cnn_module_list = net_utils.list_nngraph_modules(protos.cnn)
    for k,m in ipairs(cnn_module_list) do
      m:evaluate()
    end
  end
  protos.fcn:evaluate()
  protos.cnn_w:evaluate()
  protos.fcn_w:evaluate()
  protos.mapper:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size/2, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0, opt.net_type) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- forward the model to get loss
    local img_emb = protos.cnn:forward(data.images)
    local img_emb_w = protos.cnn_w:forward(img_emb)
    local topic_emb = protos.fcn:forward(data.topics)
    local topic_emb_w = protos.fcn_w:forward(topic_emb)
    local combine_emb = protos.combiner:forward({img_emb_w, topic_emb_w})
    local feats = protos.mapper:forward(combine_emb)
    local expanded_feats = protos.expander:forward(feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local seq = protos.lm:sample(feats, {beam_size = evalopt.beam_size})
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

local function eval_split_test2014(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
  local half_batch_size = opt.batch_size/2
  protos.cnn:evaluate()
  protos.fcn:evaluate()
  protos.cnn_w:evaluate()
  protos.fcn_w:evaluate()
  protos.mapper:evaluate()
  protos.lm:evaluate()
  loader2:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader2:getBatch{batch_size = half_batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0, opt.net_type) -- preprocess in place, and don't augment
    n = n + data.images:size(1)
    seq_list = torch.LongTensor(evalopt.topn, loader:getSeqLength(), half_batch_size)
    seqconf_list = torch.FloatTensor(evalopt.topn, half_batch_size)
    local loss = 0
    for ti = 1,evalopt.topn do
      -- forward the model to get loss
      local img_emb = protos.cnn:forward(data.images)
      local img_emb_w = protos.cnn_w:forward(img_emb)
      local topic_emb = protos.fcn:forward(data.topics[ti])
      local topic_emb_w = protos.fcn_w:forward(topic_emb)
      local combine_emb = protos.combiner:forward({img_emb_w, topic_emb_w})
      local feats = protos.mapper:forward(combine_emb)

      -- forward the model to also get generated samples for each image
      local seqx, seq_confx = protos.lm:sample(feats, {beam_size = evalopt.beam_size})
      seq_list[ti] = seqx
      lastconf = torch.FloatTensor(half_batch_size):zero()
      for bi = 1,half_batch_size do
        for si = 1,seq_confx:size(1) do
          if seq_confx[seq_confx:size(1) - si + 1][bi] > 0 then
            lastconf[bi] = seq_confx[seq_confx:size(1) - si + 1][bi]
            break
          end
        end
      end
      seqconf_list[ti] = lastconf
      --seqconf_list[ti] = seq_confx[-1]
      loss_evals = loss_evals + 1
    end
    _, seq_inds = seqconf_list:max(1)
    seq_inds = seq_inds[1]
    seq = torch.LongTensor(loader:getSeqLength(), half_batch_size):zero()
    for bi = 1,half_batch_size do
      seq[{{}, bi}] = seq_list[seq_inds[bi]][{{},bi}]
    end
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end
    if loss_evals % 10 == 0 then collectgarbage() end
    if n >= val_images_use then break end -- we've used enough images
  end
  utils.write_json('coco-caption/val2014' .. opt.id .. '.json', predictions)

  return predictions
end


local function eval_split2(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
  local half_batch_size = opt.batch_size/2
  protos.cnn:evaluate()
  protos.fcn:evaluate()
  protos.cnn_w:evaluate()
  protos.fcn_w:evaluate()
  protos.mapper:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = half_batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0, opt.net_type) -- preprocess in place, and don't augment
    n = n + data.images:size(1)
    seq_list = torch.LongTensor(evalopt.topn, loader:getSeqLength(), half_batch_size)
    seqconf_list = torch.FloatTensor(evalopt.topn, half_batch_size)
    local loss = 0
    for ti = 1,evalopt.topn do
      -- forward the model to get loss
      local img_emb = protos.cnn:forward(data.images)
      local img_emb_w = protos.cnn_w:forward(img_emb)
      local topic_emb = protos.fcn:forward(data.topics[ti])
      local topic_emb_w = protos.fcn_w:forward(topic_emb)
      local combine_emb = protos.combiner:forward({img_emb_w, topic_emb_w})
      local feats = protos.mapper:forward(combine_emb)
      local expanded_feats = protos.expander:forward(feats)
      local logprobs = protos.lm:forward{expanded_feats, data.labels}
      loss = protos.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss
      loss_evals = loss_evals + 1

      -- forward the model to also get generated samples for each image
      local seqx, seq_confx = protos.lm:sample(feats, {beam_size = evalopt.beam_size})
      seq_list[ti] = seqx
      lastconf = torch.FloatTensor(half_batch_size):zero()
      for bi = 1,half_batch_size do
        for si = 1,seq_confx:size(1) do
          if seq_confx[seq_confx:size(1) - si + 1][bi] > 0 then
            lastconf[bi] = seq_confx[seq_confx:size(1) - si + 1][bi]
            break
          end
        end
      end
      seqconf_list[ti] = lastconf
      --seqconf_list[ti] = seq_confx[-1]
    end
    _, seq_inds = seqconf_list:max(1)
    seq_inds = seq_inds[1]
    seq = torch.LongTensor(loader:getSeqLength(), half_batch_size):zero()
    for bi = 1,half_batch_size do
      seq[{{}, bi}] = seq_list[seq_inds[bi]][{{},bi}]
    end
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

local function eval_getcaptions(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
  local half_batch_size = opt.batch_size/2
  
  if opt.net_type == 1 then
    protos.cnn:evaluate()
  else 
    cnn_module_list = net_utils.list_nngraph_modules(protos.cnn)
    for k,m in ipairs(cnn_module_list) do
      m:evaluate()
    end
  end
  protos.fcn:evaluate()
  protos.cnn_w:evaluate()
  protos.fcn_w:evaluate()
  protos.mapper:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = half_batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0, opt.net_type) -- preprocess in place, and don't augment
    n = n + data.images:size(1)
    seq_list = torch.LongTensor(loader:getSeqLength(), half_batch_size, evalopt.topn)
    seqconf_list = torch.FloatTensor(evalopt.topn, half_batch_size)
    local loss = 0
    for ti = 1,evalopt.topn do
      -- forward the model to get loss
      local img_emb = protos.cnn:forward(data.images)
      local img_emb_w = protos.cnn_w:forward(img_emb)
      local topic_emb = protos.fcn:forward(data.topics[ti])
      local topic_emb_w = protos.fcn_w:forward(topic_emb)
      local combine_emb = protos.combiner:forward({img_emb_w, topic_emb_w})
      local feats = protos.mapper:forward(combine_emb)
      local expanded_feats = protos.expander:forward(feats)
      local logprobs = protos.lm:forward{expanded_feats, data.labels}
      loss = protos.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss
      loss_evals = loss_evals + 1

      -- forward the model to also get generated samples for each image
      local seq = protos.lm:sample(feats, {beam_size = evalopt.beam_size})
      seq_list[{{}, {}, ti}] = seq
    end
    seq_list = seq_list:resize(seq_list:size(1), seq_list:size(2)*seq_list:size(3))
    print(seq_list:size())
    local sents = net_utils.decode_sequence(vocab, seq_list)
    for k1=1,half_batch_size do
      for k2=1,evalopt.topn do
        local entry = {image_id = data.infos[k1].id, caption = sents[k2 + (k1-1)*evalopt.topn]}
        table.insert(predictions, entry)
        if verbose then
          _, tmp = data.topics[k2][k1]:sort()
          topic_id = tmp[-1]
          print(string.format('image %s, topic %d: %s', entry.image_id, topic_id, entry.caption))
        end
      end
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  return loss_sum/loss_evals, predictions
end


-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  if opt.net_type == 1 then
    protos.cnn:training()
  end
  --protos.cnn:training()
  --cnn_module_list = net_utils.list_nngraph_modules(protos.cnn)
  --for k,m in ipairs(cnn_module_list) do
  --  m:training()
  --end
  protos.fcn:training()
  protos.cnn_w:training()
  protos.fcn_w:training()
  protos.mapper:training()
  protos.lm:training()
  grad_params:zero()
  map_grad_params:zero()
  cnn_w_grad_params:zero()
  fcn_w_grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
    fcn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  --print(data.images:size(), data.labels:size(), data.topics:size())
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0, opt.net_type) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  local img_emb = protos.cnn:forward(data.images)
  --local expanded_img_emb = protos.expander:forward(img_emb)
  --local img_emb_w = protos.cnn_w:forward(expanded_img_emb)
  local img_emb_w = protos.cnn_w:forward(img_emb)
  local topic_emb = protos.fcn:forward(data.topics)
  local topic_emb_w = protos.fcn_w:forward(topic_emb)
  local combine_emb = protos.combiner:forward({img_emb_w, topic_emb_w})
  local feats = protos.mapper:forward(combine_emb)

  -- we have to expand out image features, once for each sentence
  local expanded_feats = protos.expander:forward(feats)
  -- forward the language model
  local logprobs = protos.lm:forward{expanded_feats, data.labels}
  --local logprobs = protos.lm:forward{feats, data.labels}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.labels)
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop language model
  local dexpanded_feats, ddummy = unpack(protos.lm:backward({expanded_feats, data.labels}, dlogprobs))
  --local dfeats, ddummy = unpack(protos.lm:backward({feats, data.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  local dfeats = protos.expander:backward(feats, dexpanded_feats)
  local dcombine_emb = protos.mapper:backward(combine_emb, dfeats)
  local dimg_emb_w, dtopic_emb_w = unpack(protos.combiner:backward({img_emb_w, topic_emb_w}, dcombine_emb))
  local dimg_emb = protos.cnn_w:backward(img_emb, dimg_emb_w)
  --local dexpanded_img_emb = protos.cnn_w:backward(expanded_img_emb, dimg_emb_w)
  --local dimg_emb = protos.cnn_w:backward(img_emb, dexpanded_img_emb)

  local dtopic_emb = protos.fcn_w:backward(topic_emb, dtopic_emb_w)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dimg = protos.cnn:backward(data.images, dimg_emb)
    local dtopic = protos.fcn:backward(data.topics, dtopic_emb)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local map_optim_state = {}
local cnn_optim_state = {}
local fcn_optim_state = {}
local cnn_w_optim_state = {}
local fcn_w_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

while true do  
  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    --local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use, beam_size=1})
    local val_loss, val_predictions, lang_stats = eval_split2('test', {val_images_use = 1000, beam_size = opt.beam_size, topn=1})
    --local val_predictions = eval_split_test2014('val2014', {val_images_use = opt.val_images_use, beam_size = opt.beam_size, topn=1})
    --local val_loss, val_predictions, lang_stats = eval_getcaptions('test', {val_images_use = 5000, beam_size = opt.beam_size, topn=5})
    --break
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        if opt.net_type == 2 then
          cnn_module_list = net_utils.list_nngraph_modules(protos.cnn)
          raw_cnn_module_list = net_utils.list_nngraph_modules(cnn_module_list[2])
          thin_cnn_module_list = net_utils.list_nngraph_modules(thin_cnn)
          thin_raw_cnn_module_list = net_utils.list_nngraph_modules(thin_cnn_module_list[2])
          for k,m in ipairs(raw_cnn_module_list) do
            if m.running_mean then
              --print(m, m.running_mean:size(), m.running_var:size())
              thin_raw_cnn_module_list[k].running_mean = m.running_mean
              thin_raw_cnn_module_list[k].running_var = m.running_var
            end
          end
        end
        save_protos.cnn = thin_cnn
        save_protos.fcn = thin_fcn
        save_protos.cnn_w = thin_cnn_w
        save_protos.fcn_w = thin_fcn_w
        save_protos.mapper = thin_map
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  
  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then 
    loss_history[iter] = losses.total_loss
    print(string.format('iter %d: %f', iter, losses.total_loss))
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  local fcn_learning_rate = opt.fcn_learning_rate
  local cnn_fcn_w_learning_rate = opt.cnn_fcn_w_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every + 1
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
    cnn_fcn_w_learning_rate = cnn_fcn_w_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
    rmsprop(map_params, map_grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, map_optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
    adagrad(map_params, map_grad_params, learning_rate, opt.optim_epsilon, map_optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
    sgd(map_params, map_grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    sgdm(map_params, map_grad_params, learning_rate, opt.optim_alpha, map_optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    sgdmom(map_params, map_grad_params, learning_rate, opt.optim_alpha, map_optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
    adam(map_params, map_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, map_optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
      --sgd(fcn_params, fcn_grad_params, fcn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
      --sgdm(fcn_params, fcn_grad_params, fcn_learning_rate, opt.fcn_optim_alpha, fcn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
      --adam(fcn_params, fcn_grad_params, fcn_learning_rate, opt.fcn_optim_alpha, opt.fcn_optim_beta, opt.optim_epsilon, fcn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end
  
  -- do cnn_w and fcn_w update
  if opt.finetune_cnn_fcn_w_after >= 0 and iter >= opt.finetune_cnn_fcn_w_after then
    if opt.cnn_fcn_w_optim == 'sgd' then
      sgd(cnn_w_params, cnn_w_grad_params, cnn_fcn_w_learning_rate)
      sgd(fcn_w_params, fcn_w_grad_params, cnn_fcn_w_learning_rate)
    elseif opt.cnn_fcn_w_optim == 'sgdm' then
      sgdm(cnn_w_params, cnn_w_grad_params, cnn_fcn_w_learning_rate, opt.cnn_fcn_w_optim_alpha, cnn_w_optim_state)
      sgdm(fcn_w_params, fcn_w_grad_params, cnn_fcn_w_learning_rate, opt.cnn_fcn_w_optim_alpha, fcn_w_optim_state)
    elseif opt.cnn_fcn_w_optim == 'adam' then
      adam(cnn_w_params, cnn_w_grad_params, cnn_fcn_w_learning_rate, opt.cnn_fcn_w_optim_alpha, opt.cnn_fcn_w_optim_beta, opt.optim_epsilon, cnn_w_optim_state)
      adam(fcn_w_params, fcn_w_grad_params, cnn_fcn_w_learning_rate, opt.cnn_fcn_w_optim_alpha, opt.cnn_fcn_w_optim_beta, opt.optim_epsilon, fcn_w_optim_state)
    else
      error('for opt.cnn_optim')
    end
  end
  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
