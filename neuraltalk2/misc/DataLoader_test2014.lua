require 'hdf5'
require 'math'
local utils = require 'misc.utils'

local DataLoader_test2014 = torch.class('DataLoader_test2014')

function DataLoader_test2014:__init(opt)
  self.topic_encode = torch.eye(opt.topic_num):float() 
  -- load the json file which contains additional information about the dataset
  print('DataLoader_test2014 loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  
  -- open the hdf5 files
  print('DataLoader_test2014 loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  print('DataLoader_test2014 loading h5 topics file: ', opt.h5_file_topics)
  self.h5_file_topics = hdf5.open(opt.h5_file_topics, 'r')
 
  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 4, '/images should be a 4D tensor')
  assert(images_size[3] == images_size[4], 'width and height must match')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.max_image_size = images_size[3]
  print(string.format('read %d images of size %dx%dx%d', self.num_images, 
            self.num_channels, self.max_image_size, self.max_image_size))

  -- load topics 
  self.im_top10topics = self.h5_file_topics:read('/im_top10topics'):all()
  -- separate out indexes for each of the provided splits
  self.split_ix = {}
  self.iterators = {}
  self.shuffle_index = {}
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end
end

function DataLoader_test2014:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader_test2014:getVocabSize()
  return self.vocab_size
end

function DataLoader_test2014:getVocab()
  return self.ix_to_word
end

function DataLoader_test2014:getSeqLength()
  return self.seq_length
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the images
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader_test2014:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 5) -- number of sequences to return per image

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local topic_batch = torch.LongTensor(batch_size, 10)
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i=1,batch_size do
    ::batch_begin::
    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then 
      ri_next = 1; 
      wrapped = true 
    end -- wrap back around
    self.iterators[split] = ri_next
    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
   
    -- fetch the sequence labels
    topic_batch[i] = self.im_top10topics[ix]
    -- fetch the image from h5
    local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            {1,self.max_image_size},{1,self.max_image_size})
    img_batch_raw[i] = img
    
    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.info.images[ix].id
    table.insert(infos, info_struct)
  end
  topic_vectors = torch.FloatTensor(10, batch_size, self.topic_encode:size(1))
  for ti = 1,10 do
    topic_vectors[ti] = self.topic_encode:index(1, topic_batch[{{}, ti}])
  end
  local data = {}
  data.images = img_batch_raw:cuda()
  data.topics = topic_vectors:cuda()
  data.infos = infos
  return data
end

