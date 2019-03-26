require 'hdf5'
require 'math'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  self.topic_encode = torch.eye(opt.topic_num):float() 
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
  print('vocab size is ' .. self.vocab_size)
  
  -- open the hdf5 files
  print('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  print('DataLoader loading h5 topics file: ', opt.h5_file_topics)
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

  -- load in the sequence data
  local seq_size = self.h5_file:read('/labels'):dataspaceSize()
  self.seq_length = seq_size[2]
  print('max sequence length in data is ' .. self.seq_length)
  -- load the pointers in full to RAM (should be small enough)
  self.label_start_ix = self.h5_file:read('/label_start_ix'):all()
  self.label_end_ix = self.h5_file:read('/label_end_ix'):all()
  
  -- load topics 
  self.im_top10topics = self.h5_file_topics:read('/im_top10topics'):all()
  self.cap_top10topics = self.h5_file_topics:read('/cap_top10topics'):all()
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
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
    self.shuffle_index[k] = torch.randperm(#v)
  end
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
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
function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 5) -- number of sequences to return per image

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  if split == 'val' or split == 'train' then
    topic_batch = torch.LongTensor(batch_size)
  elseif split == 'test' then
    topic_batch = torch.LongTensor(batch_size, 10)
  else
    topic_batch = torch.LongTensor(batch_size * seq_per_img)
  end
  local label_batch = torch.LongTensor(batch_size * seq_per_img, self.seq_length)
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
      self.shuffle_index[split] = torch.randperm(self.shuffle_index[split]:size(1))
    end -- wrap back around
    self.iterators[split] = ri_next
    if split == 'val' or split == 'test' then
      ix = split_ix[ri]
    else 
      --ix = split_ix[self.shuffle_index[split][ri]]
      ix = split_ix[ri]
    end
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
   
    -- fetch the sequence labels
    local ix1 = self.label_start_ix[ix]
    local ix2 = self.label_end_ix[ix]
    local ncap = ix2 - ix1 + 1 -- number of captions available for this image
    assert(ncap > 0 and ncap == seq_per_img, 'an image does not have any label. this can be handled but right now isn\'t')
    if split == 'train' then
      local try_times = 0
      local valid_p = {}
      local valid_count = 0
      repeat
        -- fetch image topic and sample one
        topic_batch[i] = self.im_top10topics[ix][math.random(1,5)]
        -- sample label with right topic
        valid_p = {}
        valid_count = 0
        for q=ix1,ix2 do
          local cap_top5topic = self.cap_top10topics[q][{{1,5}}]
          for p=1, 5 do
            if cap_top5topic[p] == topic_batch[i] then
              valid_count = valid_count + 1
              valid_p[valid_count] = q
              break
            end
          end
        end
        try_times = try_times + 1
      until(valid_count >= 2 or try_times > 10)
      if valid_count < 2 then 
        goto batch_begin
      end
      
      -- fetch the image from h5
      local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                              {1,self.max_image_size},{1,self.max_image_size})
      img_batch_raw[i] = img
      
      -- sample labels
      local seq
      seq = torch.LongTensor(seq_per_img, self.seq_length)
      for q=1, seq_per_img do
        local il = torch.random(1,valid_count)
        local ixl = valid_p[il]
        seq[{ {q,q} }] = self.h5_file:read('/labels'):partial({ixl, ixl}, {1,self.seq_length})
      end

      local il = (i-1)*seq_per_img+1
      label_batch[{ {il,il+seq_per_img-1} }] = seq
      -- and record associated info as well
      local info_struct = {}
      info_struct.id = self.info.images[ix].id
      info_struct.file_path = self.info.images[ix].file_path
      table.insert(infos, info_struct)
    elseif split == 'train1' then
      local il = (i-1)*seq_per_img+1
      for q=ix1,ix2 do
        topic_batch[il + q - ix1] = self.cap_top10topics[q][1]
      end
      -- fetch the image from h5
      local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                              {1,self.max_image_size},{1,self.max_image_size})
      img_batch_raw[i] = img
      
      local seq = self.h5_file:read('/labels'):partial({ix1, ix2}, {1, self.seq_length})
      local il = (i-1)*seq_per_img+1
      label_batch[{ {il,il+seq_per_img-1} }] = seq
      -- and record associated info as well
      local info_struct = {}
      info_struct.id = self.info.images[ix].id
      info_struct.file_path = self.info.images[ix].file_path
      table.insert(infos, info_struct)
    elseif split == 'val' or split == 'test' then
      if split == 'val' then 
        topic_batch[i] = self.im_top10topics[ix][1]
      else 
        topic_batch[i] = self.im_top10topics[ix]
      end
      -- fetch the image from h5
      local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                              {1,self.max_image_size},{1,self.max_image_size})
      img_batch_raw[i] = img
      
      local seq = self.h5_file:read('/labels'):partial({ix1, ix2}, {1, self.seq_length})
      local il = (i-1)*seq_per_img+1
      label_batch[{ {il,il+seq_per_img-1} }] = seq
      -- and record associated info as well
      local info_struct = {}
      info_struct.id = self.info.images[ix].id
      info_struct.file_path = self.info.images[ix].file_path
      table.insert(infos, info_struct)
    else 
      print('unrecognize split...')
    end
  end
  if split == 'test' then
    topic_vectors = torch.FloatTensor(10, batch_size, self.topic_encode:size(1))
    for ti = 1,10 do
      topic_vectors[ti] = self.topic_encode:index(1, topic_batch[{{}, ti}])
    end
  else
    topic_vectors = self.topic_encode:index(1, topic_batch)
  end
  local data = {}
  data.images = img_batch_raw:cuda()
  data.topics = topic_vectors:cuda()
  data.labels = label_batch:transpose(1,2):contiguous():cuda() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos
  return data
end

