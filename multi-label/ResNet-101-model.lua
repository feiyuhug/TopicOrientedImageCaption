require 'nngraph'

modmap = {}

data = nn.Identity()()
modmap[#modmap+1] = {data}

conv1 = nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)(data)
modmap[#modmap+1] = {conv1}

bn_conv1 = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(conv1)
modmap[#modmap+1] = {bn_conv1}

scale_conv1_scale = nn.CMul(1, 64, 1, 1)(bn_conv1)
scale_conv1 = nn.Add(1)(scale_conv1_scale)
modmap[#modmap+1] = {scale_conv1_scale, scale_conv1}

conv1_relu = nn.ReLU(true)(scale_conv1)
modmap[#modmap+1] = {conv1_relu}

pool1 = nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil()(conv1_relu)
modmap[#modmap+1] = {pool1}

res2a_branch1 = nn.SpatialConvolution(64, 256, 1, 1, 1, 1, 0, 0)(pool1)
modmap[#modmap+1] = {res2a_branch1}

bn2a_branch1 = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res2a_branch1)
modmap[#modmap+1] = {bn2a_branch1}

scale2a_branch1_scale = nn.CMul(1, 256, 1, 1)(bn2a_branch1)
scale2a_branch1 = nn.Add(1)(scale2a_branch1_scale)
modmap[#modmap+1] = {scale2a_branch1_scale, scale2a_branch1}

res2a_branch2a = nn.SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0)(pool1)
modmap[#modmap+1] = {res2a_branch2a}

bn2a_branch2a = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(res2a_branch2a)
modmap[#modmap+1] = {bn2a_branch2a}

scale2a_branch2a_scale = nn.CMul(1, 64, 1, 1)(bn2a_branch2a)
scale2a_branch2a = nn.Add(1)(scale2a_branch2a_scale)
modmap[#modmap+1] = {scale2a_branch2a_scale, scale2a_branch2a}

res2a_branch2a_relu = nn.ReLU(true)(scale2a_branch2a)
modmap[#modmap+1] = {res2a_branch2a_relu}

res2a_branch2b = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(res2a_branch2a_relu)
modmap[#modmap+1] = {res2a_branch2b}

bn2a_branch2b = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(res2a_branch2b)
modmap[#modmap+1] = {bn2a_branch2b}

scale2a_branch2b_scale = nn.CMul(1, 64, 1, 1)(bn2a_branch2b)
scale2a_branch2b = nn.Add(1)(scale2a_branch2b_scale)
modmap[#modmap+1] = {scale2a_branch2b_scale, scale2a_branch2b}

res2a_branch2b_relu = nn.ReLU(true)(scale2a_branch2b)
modmap[#modmap+1] = {res2a_branch2b_relu}

res2a_branch2c = nn.SpatialConvolution(64, 256, 1, 1, 1, 1, 0, 0)(res2a_branch2b_relu)
modmap[#modmap+1] = {res2a_branch2c}

bn2a_branch2c = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res2a_branch2c)
modmap[#modmap+1] = {bn2a_branch2c}

scale2a_branch2c_scale = nn.CMul(1, 256, 1, 1)(bn2a_branch2c)
scale2a_branch2c = nn.Add(1)(scale2a_branch2c_scale)
modmap[#modmap+1] = {scale2a_branch2c_scale, scale2a_branch2c}

res2a = nn.CAddTable()({scale2a_branch1, scale2a_branch2c})
modmap[#modmap+1] = {res2a}

res2a_relu = nn.ReLU(true)(res2a)
modmap[#modmap+1] = {res2a_relu}

res2b_branch2a = nn.SpatialConvolution(256, 64, 1, 1, 1, 1, 0, 0)(res2a_relu)
modmap[#modmap+1] = {res2b_branch2a}

bn2b_branch2a = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(res2b_branch2a)
modmap[#modmap+1] = {bn2b_branch2a}

scale2b_branch2a_scale = nn.CMul(1, 64, 1, 1)(bn2b_branch2a)
scale2b_branch2a = nn.Add(1)(scale2b_branch2a_scale)
modmap[#modmap+1] = {scale2b_branch2a_scale, scale2b_branch2a}

res2b_branch2a_relu = nn.ReLU(true)(scale2b_branch2a)
modmap[#modmap+1] = {res2b_branch2a_relu}

res2b_branch2b = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(res2b_branch2a_relu)
modmap[#modmap+1] = {res2b_branch2b}

bn2b_branch2b = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(res2b_branch2b)
modmap[#modmap+1] = {bn2b_branch2b}

scale2b_branch2b_scale = nn.CMul(1, 64, 1, 1)(bn2b_branch2b)
scale2b_branch2b = nn.Add(1)(scale2b_branch2b_scale)
modmap[#modmap+1] = {scale2b_branch2b_scale, scale2b_branch2b}

res2b_branch2b_relu = nn.ReLU(true)(scale2b_branch2b)
modmap[#modmap+1] = {res2b_branch2b_relu}

res2b_branch2c = nn.SpatialConvolution(64, 256, 1, 1, 1, 1, 0, 0)(res2b_branch2b_relu)
modmap[#modmap+1] = {res2b_branch2c}

bn2b_branch2c = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res2b_branch2c)
modmap[#modmap+1] = {bn2b_branch2c}

scale2b_branch2c_scale = nn.CMul(1, 256, 1, 1)(bn2b_branch2c)
scale2b_branch2c = nn.Add(1)(scale2b_branch2c_scale)
modmap[#modmap+1] = {scale2b_branch2c_scale, scale2b_branch2c}

res2b = nn.CAddTable()({res2a_relu, scale2b_branch2c})
modmap[#modmap+1] = {res2b}

res2b_relu = nn.ReLU(true)(res2b)
modmap[#modmap+1] = {res2b_relu}

res2c_branch2a = nn.SpatialConvolution(256, 64, 1, 1, 1, 1, 0, 0)(res2b_relu)
modmap[#modmap+1] = {res2c_branch2a}

bn2c_branch2a = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(res2c_branch2a)
modmap[#modmap+1] = {bn2c_branch2a}

scale2c_branch2a_scale = nn.CMul(1, 64, 1, 1)(bn2c_branch2a)
scale2c_branch2a = nn.Add(1)(scale2c_branch2a_scale)
modmap[#modmap+1] = {scale2c_branch2a_scale, scale2c_branch2a}

res2c_branch2a_relu = nn.ReLU(true)(scale2c_branch2a)
modmap[#modmap+1] = {res2c_branch2a_relu}

res2c_branch2b = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(res2c_branch2a_relu)
modmap[#modmap+1] = {res2c_branch2b}

bn2c_branch2b = nn.SpatialBatchNormalization(64, 1e-05, 0.999)(res2c_branch2b)
modmap[#modmap+1] = {bn2c_branch2b}

scale2c_branch2b_scale = nn.CMul(1, 64, 1, 1)(bn2c_branch2b)
scale2c_branch2b = nn.Add(1)(scale2c_branch2b_scale)
modmap[#modmap+1] = {scale2c_branch2b_scale, scale2c_branch2b}

res2c_branch2b_relu = nn.ReLU(true)(scale2c_branch2b)
modmap[#modmap+1] = {res2c_branch2b_relu}

res2c_branch2c = nn.SpatialConvolution(64, 256, 1, 1, 1, 1, 0, 0)(res2c_branch2b_relu)
modmap[#modmap+1] = {res2c_branch2c}

bn2c_branch2c = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res2c_branch2c)
modmap[#modmap+1] = {bn2c_branch2c}

scale2c_branch2c_scale = nn.CMul(1, 256, 1, 1)(bn2c_branch2c)
scale2c_branch2c = nn.Add(1)(scale2c_branch2c_scale)
modmap[#modmap+1] = {scale2c_branch2c_scale, scale2c_branch2c}

res2c = nn.CAddTable()({res2b_relu, scale2c_branch2c})
modmap[#modmap+1] = {res2c}

res2c_relu = nn.ReLU(true)(res2c)
modmap[#modmap+1] = {res2c_relu}

res3a_branch1 = nn.SpatialConvolution(256, 512, 1, 1, 2, 2, 0, 0)(res2c_relu)
modmap[#modmap+1] = {res3a_branch1}

bn3a_branch1 = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res3a_branch1)
modmap[#modmap+1] = {bn3a_branch1}

scale3a_branch1_scale = nn.CMul(1, 512, 1, 1)(bn3a_branch1)
scale3a_branch1 = nn.Add(1)(scale3a_branch1_scale)
modmap[#modmap+1] = {scale3a_branch1_scale, scale3a_branch1}

res3a_branch2a = nn.SpatialConvolution(256, 128, 1, 1, 2, 2, 0, 0)(res2c_relu)
modmap[#modmap+1] = {res3a_branch2a}

bn3a_branch2a = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3a_branch2a)
modmap[#modmap+1] = {bn3a_branch2a}

scale3a_branch2a_scale = nn.CMul(1, 128, 1, 1)(bn3a_branch2a)
scale3a_branch2a = nn.Add(1)(scale3a_branch2a_scale)
modmap[#modmap+1] = {scale3a_branch2a_scale, scale3a_branch2a}

res3a_branch2a_relu = nn.ReLU(true)(scale3a_branch2a)
modmap[#modmap+1] = {res3a_branch2a_relu}

res3a_branch2b = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(res3a_branch2a_relu)
modmap[#modmap+1] = {res3a_branch2b}

bn3a_branch2b = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3a_branch2b)
modmap[#modmap+1] = {bn3a_branch2b}

scale3a_branch2b_scale = nn.CMul(1, 128, 1, 1)(bn3a_branch2b)
scale3a_branch2b = nn.Add(1)(scale3a_branch2b_scale)
modmap[#modmap+1] = {scale3a_branch2b_scale, scale3a_branch2b}

res3a_branch2b_relu = nn.ReLU(true)(scale3a_branch2b)
modmap[#modmap+1] = {res3a_branch2b_relu}

res3a_branch2c = nn.SpatialConvolution(128, 512, 1, 1, 1, 1, 0, 0)(res3a_branch2b_relu)
modmap[#modmap+1] = {res3a_branch2c}

bn3a_branch2c = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res3a_branch2c)
modmap[#modmap+1] = {bn3a_branch2c}

scale3a_branch2c_scale = nn.CMul(1, 512, 1, 1)(bn3a_branch2c)
scale3a_branch2c = nn.Add(1)(scale3a_branch2c_scale)
modmap[#modmap+1] = {scale3a_branch2c_scale, scale3a_branch2c}

res3a = nn.CAddTable()({scale3a_branch1, scale3a_branch2c})
modmap[#modmap+1] = {res3a}

res3a_relu = nn.ReLU(true)(res3a)
modmap[#modmap+1] = {res3a_relu}

res3b1_branch2a = nn.SpatialConvolution(512, 128, 1, 1, 1, 1, 0, 0)(res3a_relu)
modmap[#modmap+1] = {res3b1_branch2a}

bn3b1_branch2a = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3b1_branch2a)
modmap[#modmap+1] = {bn3b1_branch2a}

scale3b1_branch2a_scale = nn.CMul(1, 128, 1, 1)(bn3b1_branch2a)
scale3b1_branch2a = nn.Add(1)(scale3b1_branch2a_scale)
modmap[#modmap+1] = {scale3b1_branch2a_scale, scale3b1_branch2a}

res3b1_branch2a_relu = nn.ReLU(true)(scale3b1_branch2a)
modmap[#modmap+1] = {res3b1_branch2a_relu}

res3b1_branch2b = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(res3b1_branch2a_relu)
modmap[#modmap+1] = {res3b1_branch2b}

bn3b1_branch2b = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3b1_branch2b)
modmap[#modmap+1] = {bn3b1_branch2b}

scale3b1_branch2b_scale = nn.CMul(1, 128, 1, 1)(bn3b1_branch2b)
scale3b1_branch2b = nn.Add(1)(scale3b1_branch2b_scale)
modmap[#modmap+1] = {scale3b1_branch2b_scale, scale3b1_branch2b}

res3b1_branch2b_relu = nn.ReLU(true)(scale3b1_branch2b)
modmap[#modmap+1] = {res3b1_branch2b_relu}

res3b1_branch2c = nn.SpatialConvolution(128, 512, 1, 1, 1, 1, 0, 0)(res3b1_branch2b_relu)
modmap[#modmap+1] = {res3b1_branch2c}

bn3b1_branch2c = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res3b1_branch2c)
modmap[#modmap+1] = {bn3b1_branch2c}

scale3b1_branch2c_scale = nn.CMul(1, 512, 1, 1)(bn3b1_branch2c)
scale3b1_branch2c = nn.Add(1)(scale3b1_branch2c_scale)
modmap[#modmap+1] = {scale3b1_branch2c_scale, scale3b1_branch2c}

res3b1 = nn.CAddTable()({res3a_relu, scale3b1_branch2c})
modmap[#modmap+1] = {res3b1}

res3b1_relu = nn.ReLU(true)(res3b1)
modmap[#modmap+1] = {res3b1_relu}

res3b2_branch2a = nn.SpatialConvolution(512, 128, 1, 1, 1, 1, 0, 0)(res3b1_relu)
modmap[#modmap+1] = {res3b2_branch2a}

bn3b2_branch2a = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3b2_branch2a)
modmap[#modmap+1] = {bn3b2_branch2a}

scale3b2_branch2a_scale = nn.CMul(1, 128, 1, 1)(bn3b2_branch2a)
scale3b2_branch2a = nn.Add(1)(scale3b2_branch2a_scale)
modmap[#modmap+1] = {scale3b2_branch2a_scale, scale3b2_branch2a}

res3b2_branch2a_relu = nn.ReLU(true)(scale3b2_branch2a)
modmap[#modmap+1] = {res3b2_branch2a_relu}

res3b2_branch2b = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(res3b2_branch2a_relu)
modmap[#modmap+1] = {res3b2_branch2b}

bn3b2_branch2b = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3b2_branch2b)
modmap[#modmap+1] = {bn3b2_branch2b}

scale3b2_branch2b_scale = nn.CMul(1, 128, 1, 1)(bn3b2_branch2b)
scale3b2_branch2b = nn.Add(1)(scale3b2_branch2b_scale)
modmap[#modmap+1] = {scale3b2_branch2b_scale, scale3b2_branch2b}

res3b2_branch2b_relu = nn.ReLU(true)(scale3b2_branch2b)
modmap[#modmap+1] = {res3b2_branch2b_relu}

res3b2_branch2c = nn.SpatialConvolution(128, 512, 1, 1, 1, 1, 0, 0)(res3b2_branch2b_relu)
modmap[#modmap+1] = {res3b2_branch2c}

bn3b2_branch2c = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res3b2_branch2c)
modmap[#modmap+1] = {bn3b2_branch2c}

scale3b2_branch2c_scale = nn.CMul(1, 512, 1, 1)(bn3b2_branch2c)
scale3b2_branch2c = nn.Add(1)(scale3b2_branch2c_scale)
modmap[#modmap+1] = {scale3b2_branch2c_scale, scale3b2_branch2c}

res3b2 = nn.CAddTable()({res3b1_relu, scale3b2_branch2c})
modmap[#modmap+1] = {res3b2}

res3b2_relu = nn.ReLU(true)(res3b2)
modmap[#modmap+1] = {res3b2_relu}

res3b3_branch2a = nn.SpatialConvolution(512, 128, 1, 1, 1, 1, 0, 0)(res3b2_relu)
modmap[#modmap+1] = {res3b3_branch2a}

bn3b3_branch2a = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3b3_branch2a)
modmap[#modmap+1] = {bn3b3_branch2a}

scale3b3_branch2a_scale = nn.CMul(1, 128, 1, 1)(bn3b3_branch2a)
scale3b3_branch2a = nn.Add(1)(scale3b3_branch2a_scale)
modmap[#modmap+1] = {scale3b3_branch2a_scale, scale3b3_branch2a}

res3b3_branch2a_relu = nn.ReLU(true)(scale3b3_branch2a)
modmap[#modmap+1] = {res3b3_branch2a_relu}

res3b3_branch2b = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(res3b3_branch2a_relu)
modmap[#modmap+1] = {res3b3_branch2b}

bn3b3_branch2b = nn.SpatialBatchNormalization(128, 1e-05, 0.999)(res3b3_branch2b)
modmap[#modmap+1] = {bn3b3_branch2b}

scale3b3_branch2b_scale = nn.CMul(1, 128, 1, 1)(bn3b3_branch2b)
scale3b3_branch2b = nn.Add(1)(scale3b3_branch2b_scale)
modmap[#modmap+1] = {scale3b3_branch2b_scale, scale3b3_branch2b}

res3b3_branch2b_relu = nn.ReLU(true)(scale3b3_branch2b)
modmap[#modmap+1] = {res3b3_branch2b_relu}

res3b3_branch2c = nn.SpatialConvolution(128, 512, 1, 1, 1, 1, 0, 0)(res3b3_branch2b_relu)
modmap[#modmap+1] = {res3b3_branch2c}

bn3b3_branch2c = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res3b3_branch2c)
modmap[#modmap+1] = {bn3b3_branch2c}

scale3b3_branch2c_scale = nn.CMul(1, 512, 1, 1)(bn3b3_branch2c)
scale3b3_branch2c = nn.Add(1)(scale3b3_branch2c_scale)
modmap[#modmap+1] = {scale3b3_branch2c_scale, scale3b3_branch2c}

res3b3 = nn.CAddTable()({res3b2_relu, scale3b3_branch2c})
modmap[#modmap+1] = {res3b3}

res3b3_relu = nn.ReLU(true)(res3b3)
modmap[#modmap+1] = {res3b3_relu}

res4a_branch1 = nn.SpatialConvolution(512, 1024, 1, 1, 2, 2, 0, 0)(res3b3_relu)
modmap[#modmap+1] = {res4a_branch1}

bn4a_branch1 = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4a_branch1)
modmap[#modmap+1] = {bn4a_branch1}

scale4a_branch1_scale = nn.CMul(1, 1024, 1, 1)(bn4a_branch1)
scale4a_branch1 = nn.Add(1)(scale4a_branch1_scale)
modmap[#modmap+1] = {scale4a_branch1_scale, scale4a_branch1}

res4a_branch2a = nn.SpatialConvolution(512, 256, 1, 1, 2, 2, 0, 0)(res3b3_relu)
modmap[#modmap+1] = {res4a_branch2a}

bn4a_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4a_branch2a)
modmap[#modmap+1] = {bn4a_branch2a}

scale4a_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4a_branch2a)
scale4a_branch2a = nn.Add(1)(scale4a_branch2a_scale)
modmap[#modmap+1] = {scale4a_branch2a_scale, scale4a_branch2a}

res4a_branch2a_relu = nn.ReLU(true)(scale4a_branch2a)
modmap[#modmap+1] = {res4a_branch2a_relu}

res4a_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4a_branch2a_relu)
modmap[#modmap+1] = {res4a_branch2b}

bn4a_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4a_branch2b)
modmap[#modmap+1] = {bn4a_branch2b}

scale4a_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4a_branch2b)
scale4a_branch2b = nn.Add(1)(scale4a_branch2b_scale)
modmap[#modmap+1] = {scale4a_branch2b_scale, scale4a_branch2b}

res4a_branch2b_relu = nn.ReLU(true)(scale4a_branch2b)
modmap[#modmap+1] = {res4a_branch2b_relu}

res4a_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4a_branch2b_relu)
modmap[#modmap+1] = {res4a_branch2c}

bn4a_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4a_branch2c)
modmap[#modmap+1] = {bn4a_branch2c}

scale4a_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4a_branch2c)
scale4a_branch2c = nn.Add(1)(scale4a_branch2c_scale)
modmap[#modmap+1] = {scale4a_branch2c_scale, scale4a_branch2c}

res4a = nn.CAddTable()({scale4a_branch1, scale4a_branch2c})
modmap[#modmap+1] = {res4a}

res4a_relu = nn.ReLU(true)(res4a)
modmap[#modmap+1] = {res4a_relu}

res4b1_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4a_relu)
modmap[#modmap+1] = {res4b1_branch2a}

bn4b1_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b1_branch2a)
modmap[#modmap+1] = {bn4b1_branch2a}

scale4b1_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b1_branch2a)
scale4b1_branch2a = nn.Add(1)(scale4b1_branch2a_scale)
modmap[#modmap+1] = {scale4b1_branch2a_scale, scale4b1_branch2a}

res4b1_branch2a_relu = nn.ReLU(true)(scale4b1_branch2a)
modmap[#modmap+1] = {res4b1_branch2a_relu}

res4b1_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b1_branch2a_relu)
modmap[#modmap+1] = {res4b1_branch2b}

bn4b1_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b1_branch2b)
modmap[#modmap+1] = {bn4b1_branch2b}

scale4b1_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b1_branch2b)
scale4b1_branch2b = nn.Add(1)(scale4b1_branch2b_scale)
modmap[#modmap+1] = {scale4b1_branch2b_scale, scale4b1_branch2b}

res4b1_branch2b_relu = nn.ReLU(true)(scale4b1_branch2b)
modmap[#modmap+1] = {res4b1_branch2b_relu}

res4b1_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b1_branch2b_relu)
modmap[#modmap+1] = {res4b1_branch2c}

bn4b1_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b1_branch2c)
modmap[#modmap+1] = {bn4b1_branch2c}

scale4b1_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b1_branch2c)
scale4b1_branch2c = nn.Add(1)(scale4b1_branch2c_scale)
modmap[#modmap+1] = {scale4b1_branch2c_scale, scale4b1_branch2c}

res4b1 = nn.CAddTable()({res4a_relu, scale4b1_branch2c})
modmap[#modmap+1] = {res4b1}

res4b1_relu = nn.ReLU(true)(res4b1)
modmap[#modmap+1] = {res4b1_relu}

res4b2_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b1_relu)
modmap[#modmap+1] = {res4b2_branch2a}

bn4b2_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b2_branch2a)
modmap[#modmap+1] = {bn4b2_branch2a}

scale4b2_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b2_branch2a)
scale4b2_branch2a = nn.Add(1)(scale4b2_branch2a_scale)
modmap[#modmap+1] = {scale4b2_branch2a_scale, scale4b2_branch2a}

res4b2_branch2a_relu = nn.ReLU(true)(scale4b2_branch2a)
modmap[#modmap+1] = {res4b2_branch2a_relu}

res4b2_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b2_branch2a_relu)
modmap[#modmap+1] = {res4b2_branch2b}

bn4b2_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b2_branch2b)
modmap[#modmap+1] = {bn4b2_branch2b}

scale4b2_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b2_branch2b)
scale4b2_branch2b = nn.Add(1)(scale4b2_branch2b_scale)
modmap[#modmap+1] = {scale4b2_branch2b_scale, scale4b2_branch2b}

res4b2_branch2b_relu = nn.ReLU(true)(scale4b2_branch2b)
modmap[#modmap+1] = {res4b2_branch2b_relu}

res4b2_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b2_branch2b_relu)
modmap[#modmap+1] = {res4b2_branch2c}

bn4b2_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b2_branch2c)
modmap[#modmap+1] = {bn4b2_branch2c}

scale4b2_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b2_branch2c)
scale4b2_branch2c = nn.Add(1)(scale4b2_branch2c_scale)
modmap[#modmap+1] = {scale4b2_branch2c_scale, scale4b2_branch2c}

res4b2 = nn.CAddTable()({res4b1_relu, scale4b2_branch2c})
modmap[#modmap+1] = {res4b2}

res4b2_relu = nn.ReLU(true)(res4b2)
modmap[#modmap+1] = {res4b2_relu}

res4b3_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b2_relu)
modmap[#modmap+1] = {res4b3_branch2a}

bn4b3_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b3_branch2a)
modmap[#modmap+1] = {bn4b3_branch2a}

scale4b3_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b3_branch2a)
scale4b3_branch2a = nn.Add(1)(scale4b3_branch2a_scale)
modmap[#modmap+1] = {scale4b3_branch2a_scale, scale4b3_branch2a}

res4b3_branch2a_relu = nn.ReLU(true)(scale4b3_branch2a)
modmap[#modmap+1] = {res4b3_branch2a_relu}

res4b3_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b3_branch2a_relu)
modmap[#modmap+1] = {res4b3_branch2b}

bn4b3_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b3_branch2b)
modmap[#modmap+1] = {bn4b3_branch2b}

scale4b3_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b3_branch2b)
scale4b3_branch2b = nn.Add(1)(scale4b3_branch2b_scale)
modmap[#modmap+1] = {scale4b3_branch2b_scale, scale4b3_branch2b}

res4b3_branch2b_relu = nn.ReLU(true)(scale4b3_branch2b)
modmap[#modmap+1] = {res4b3_branch2b_relu}

res4b3_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b3_branch2b_relu)
modmap[#modmap+1] = {res4b3_branch2c}

bn4b3_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b3_branch2c)
modmap[#modmap+1] = {bn4b3_branch2c}

scale4b3_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b3_branch2c)
scale4b3_branch2c = nn.Add(1)(scale4b3_branch2c_scale)
modmap[#modmap+1] = {scale4b3_branch2c_scale, scale4b3_branch2c}

res4b3 = nn.CAddTable()({res4b2_relu, scale4b3_branch2c})
modmap[#modmap+1] = {res4b3}

res4b3_relu = nn.ReLU(true)(res4b3)
modmap[#modmap+1] = {res4b3_relu}

res4b4_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b3_relu)
modmap[#modmap+1] = {res4b4_branch2a}

bn4b4_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b4_branch2a)
modmap[#modmap+1] = {bn4b4_branch2a}

scale4b4_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b4_branch2a)
scale4b4_branch2a = nn.Add(1)(scale4b4_branch2a_scale)
modmap[#modmap+1] = {scale4b4_branch2a_scale, scale4b4_branch2a}

res4b4_branch2a_relu = nn.ReLU(true)(scale4b4_branch2a)
modmap[#modmap+1] = {res4b4_branch2a_relu}

res4b4_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b4_branch2a_relu)
modmap[#modmap+1] = {res4b4_branch2b}

bn4b4_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b4_branch2b)
modmap[#modmap+1] = {bn4b4_branch2b}

scale4b4_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b4_branch2b)
scale4b4_branch2b = nn.Add(1)(scale4b4_branch2b_scale)
modmap[#modmap+1] = {scale4b4_branch2b_scale, scale4b4_branch2b}

res4b4_branch2b_relu = nn.ReLU(true)(scale4b4_branch2b)
modmap[#modmap+1] = {res4b4_branch2b_relu}

res4b4_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b4_branch2b_relu)
modmap[#modmap+1] = {res4b4_branch2c}

bn4b4_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b4_branch2c)
modmap[#modmap+1] = {bn4b4_branch2c}

scale4b4_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b4_branch2c)
scale4b4_branch2c = nn.Add(1)(scale4b4_branch2c_scale)
modmap[#modmap+1] = {scale4b4_branch2c_scale, scale4b4_branch2c}

res4b4 = nn.CAddTable()({res4b3_relu, scale4b4_branch2c})
modmap[#modmap+1] = {res4b4}

res4b4_relu = nn.ReLU(true)(res4b4)
modmap[#modmap+1] = {res4b4_relu}

res4b5_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b4_relu)
modmap[#modmap+1] = {res4b5_branch2a}

bn4b5_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b5_branch2a)
modmap[#modmap+1] = {bn4b5_branch2a}

scale4b5_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b5_branch2a)
scale4b5_branch2a = nn.Add(1)(scale4b5_branch2a_scale)
modmap[#modmap+1] = {scale4b5_branch2a_scale, scale4b5_branch2a}

res4b5_branch2a_relu = nn.ReLU(true)(scale4b5_branch2a)
modmap[#modmap+1] = {res4b5_branch2a_relu}

res4b5_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b5_branch2a_relu)
modmap[#modmap+1] = {res4b5_branch2b}

bn4b5_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b5_branch2b)
modmap[#modmap+1] = {bn4b5_branch2b}

scale4b5_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b5_branch2b)
scale4b5_branch2b = nn.Add(1)(scale4b5_branch2b_scale)
modmap[#modmap+1] = {scale4b5_branch2b_scale, scale4b5_branch2b}

res4b5_branch2b_relu = nn.ReLU(true)(scale4b5_branch2b)
modmap[#modmap+1] = {res4b5_branch2b_relu}

res4b5_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b5_branch2b_relu)
modmap[#modmap+1] = {res4b5_branch2c}

bn4b5_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b5_branch2c)
modmap[#modmap+1] = {bn4b5_branch2c}

scale4b5_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b5_branch2c)
scale4b5_branch2c = nn.Add(1)(scale4b5_branch2c_scale)
modmap[#modmap+1] = {scale4b5_branch2c_scale, scale4b5_branch2c}

res4b5 = nn.CAddTable()({res4b4_relu, scale4b5_branch2c})
modmap[#modmap+1] = {res4b5}

res4b5_relu = nn.ReLU(true)(res4b5)
modmap[#modmap+1] = {res4b5_relu}

res4b6_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b5_relu)
modmap[#modmap+1] = {res4b6_branch2a}

bn4b6_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b6_branch2a)
modmap[#modmap+1] = {bn4b6_branch2a}

scale4b6_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b6_branch2a)
scale4b6_branch2a = nn.Add(1)(scale4b6_branch2a_scale)
modmap[#modmap+1] = {scale4b6_branch2a_scale, scale4b6_branch2a}

res4b6_branch2a_relu = nn.ReLU(true)(scale4b6_branch2a)
modmap[#modmap+1] = {res4b6_branch2a_relu}

res4b6_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b6_branch2a_relu)
modmap[#modmap+1] = {res4b6_branch2b}

bn4b6_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b6_branch2b)
modmap[#modmap+1] = {bn4b6_branch2b}

scale4b6_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b6_branch2b)
scale4b6_branch2b = nn.Add(1)(scale4b6_branch2b_scale)
modmap[#modmap+1] = {scale4b6_branch2b_scale, scale4b6_branch2b}

res4b6_branch2b_relu = nn.ReLU(true)(scale4b6_branch2b)
modmap[#modmap+1] = {res4b6_branch2b_relu}

res4b6_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b6_branch2b_relu)
modmap[#modmap+1] = {res4b6_branch2c}

bn4b6_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b6_branch2c)
modmap[#modmap+1] = {bn4b6_branch2c}

scale4b6_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b6_branch2c)
scale4b6_branch2c = nn.Add(1)(scale4b6_branch2c_scale)
modmap[#modmap+1] = {scale4b6_branch2c_scale, scale4b6_branch2c}

res4b6 = nn.CAddTable()({res4b5_relu, scale4b6_branch2c})
modmap[#modmap+1] = {res4b6}

res4b6_relu = nn.ReLU(true)(res4b6)
modmap[#modmap+1] = {res4b6_relu}

res4b7_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b6_relu)
modmap[#modmap+1] = {res4b7_branch2a}

bn4b7_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b7_branch2a)
modmap[#modmap+1] = {bn4b7_branch2a}

scale4b7_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b7_branch2a)
scale4b7_branch2a = nn.Add(1)(scale4b7_branch2a_scale)
modmap[#modmap+1] = {scale4b7_branch2a_scale, scale4b7_branch2a}

res4b7_branch2a_relu = nn.ReLU(true)(scale4b7_branch2a)
modmap[#modmap+1] = {res4b7_branch2a_relu}

res4b7_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b7_branch2a_relu)
modmap[#modmap+1] = {res4b7_branch2b}

bn4b7_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b7_branch2b)
modmap[#modmap+1] = {bn4b7_branch2b}

scale4b7_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b7_branch2b)
scale4b7_branch2b = nn.Add(1)(scale4b7_branch2b_scale)
modmap[#modmap+1] = {scale4b7_branch2b_scale, scale4b7_branch2b}

res4b7_branch2b_relu = nn.ReLU(true)(scale4b7_branch2b)
modmap[#modmap+1] = {res4b7_branch2b_relu}

res4b7_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b7_branch2b_relu)
modmap[#modmap+1] = {res4b7_branch2c}

bn4b7_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b7_branch2c)
modmap[#modmap+1] = {bn4b7_branch2c}

scale4b7_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b7_branch2c)
scale4b7_branch2c = nn.Add(1)(scale4b7_branch2c_scale)
modmap[#modmap+1] = {scale4b7_branch2c_scale, scale4b7_branch2c}

res4b7 = nn.CAddTable()({res4b6_relu, scale4b7_branch2c})
modmap[#modmap+1] = {res4b7}

res4b7_relu = nn.ReLU(true)(res4b7)
modmap[#modmap+1] = {res4b7_relu}

res4b8_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b7_relu)
modmap[#modmap+1] = {res4b8_branch2a}

bn4b8_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b8_branch2a)
modmap[#modmap+1] = {bn4b8_branch2a}

scale4b8_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b8_branch2a)
scale4b8_branch2a = nn.Add(1)(scale4b8_branch2a_scale)
modmap[#modmap+1] = {scale4b8_branch2a_scale, scale4b8_branch2a}

res4b8_branch2a_relu = nn.ReLU(true)(scale4b8_branch2a)
modmap[#modmap+1] = {res4b8_branch2a_relu}

res4b8_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b8_branch2a_relu)
modmap[#modmap+1] = {res4b8_branch2b}

bn4b8_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b8_branch2b)
modmap[#modmap+1] = {bn4b8_branch2b}

scale4b8_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b8_branch2b)
scale4b8_branch2b = nn.Add(1)(scale4b8_branch2b_scale)
modmap[#modmap+1] = {scale4b8_branch2b_scale, scale4b8_branch2b}

res4b8_branch2b_relu = nn.ReLU(true)(scale4b8_branch2b)
modmap[#modmap+1] = {res4b8_branch2b_relu}

res4b8_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b8_branch2b_relu)
modmap[#modmap+1] = {res4b8_branch2c}

bn4b8_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b8_branch2c)
modmap[#modmap+1] = {bn4b8_branch2c}

scale4b8_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b8_branch2c)
scale4b8_branch2c = nn.Add(1)(scale4b8_branch2c_scale)
modmap[#modmap+1] = {scale4b8_branch2c_scale, scale4b8_branch2c}

res4b8 = nn.CAddTable()({res4b7_relu, scale4b8_branch2c})
modmap[#modmap+1] = {res4b8}

res4b8_relu = nn.ReLU(true)(res4b8)
modmap[#modmap+1] = {res4b8_relu}

res4b9_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b8_relu)
modmap[#modmap+1] = {res4b9_branch2a}

bn4b9_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b9_branch2a)
modmap[#modmap+1] = {bn4b9_branch2a}

scale4b9_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b9_branch2a)
scale4b9_branch2a = nn.Add(1)(scale4b9_branch2a_scale)
modmap[#modmap+1] = {scale4b9_branch2a_scale, scale4b9_branch2a}

res4b9_branch2a_relu = nn.ReLU(true)(scale4b9_branch2a)
modmap[#modmap+1] = {res4b9_branch2a_relu}

res4b9_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b9_branch2a_relu)
modmap[#modmap+1] = {res4b9_branch2b}

bn4b9_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b9_branch2b)
modmap[#modmap+1] = {bn4b9_branch2b}

scale4b9_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b9_branch2b)
scale4b9_branch2b = nn.Add(1)(scale4b9_branch2b_scale)
modmap[#modmap+1] = {scale4b9_branch2b_scale, scale4b9_branch2b}

res4b9_branch2b_relu = nn.ReLU(true)(scale4b9_branch2b)
modmap[#modmap+1] = {res4b9_branch2b_relu}

res4b9_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b9_branch2b_relu)
modmap[#modmap+1] = {res4b9_branch2c}

bn4b9_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b9_branch2c)
modmap[#modmap+1] = {bn4b9_branch2c}

scale4b9_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b9_branch2c)
scale4b9_branch2c = nn.Add(1)(scale4b9_branch2c_scale)
modmap[#modmap+1] = {scale4b9_branch2c_scale, scale4b9_branch2c}

res4b9 = nn.CAddTable()({res4b8_relu, scale4b9_branch2c})
modmap[#modmap+1] = {res4b9}

res4b9_relu = nn.ReLU(true)(res4b9)
modmap[#modmap+1] = {res4b9_relu}

res4b10_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b9_relu)
modmap[#modmap+1] = {res4b10_branch2a}

bn4b10_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b10_branch2a)
modmap[#modmap+1] = {bn4b10_branch2a}

scale4b10_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b10_branch2a)
scale4b10_branch2a = nn.Add(1)(scale4b10_branch2a_scale)
modmap[#modmap+1] = {scale4b10_branch2a_scale, scale4b10_branch2a}

res4b10_branch2a_relu = nn.ReLU(true)(scale4b10_branch2a)
modmap[#modmap+1] = {res4b10_branch2a_relu}

res4b10_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b10_branch2a_relu)
modmap[#modmap+1] = {res4b10_branch2b}

bn4b10_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b10_branch2b)
modmap[#modmap+1] = {bn4b10_branch2b}

scale4b10_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b10_branch2b)
scale4b10_branch2b = nn.Add(1)(scale4b10_branch2b_scale)
modmap[#modmap+1] = {scale4b10_branch2b_scale, scale4b10_branch2b}

res4b10_branch2b_relu = nn.ReLU(true)(scale4b10_branch2b)
modmap[#modmap+1] = {res4b10_branch2b_relu}

res4b10_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b10_branch2b_relu)
modmap[#modmap+1] = {res4b10_branch2c}

bn4b10_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b10_branch2c)
modmap[#modmap+1] = {bn4b10_branch2c}

scale4b10_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b10_branch2c)
scale4b10_branch2c = nn.Add(1)(scale4b10_branch2c_scale)
modmap[#modmap+1] = {scale4b10_branch2c_scale, scale4b10_branch2c}

res4b10 = nn.CAddTable()({res4b9_relu, scale4b10_branch2c})
modmap[#modmap+1] = {res4b10}

res4b10_relu = nn.ReLU(true)(res4b10)
modmap[#modmap+1] = {res4b10_relu}

res4b11_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b10_relu)
modmap[#modmap+1] = {res4b11_branch2a}

bn4b11_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b11_branch2a)
modmap[#modmap+1] = {bn4b11_branch2a}

scale4b11_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b11_branch2a)
scale4b11_branch2a = nn.Add(1)(scale4b11_branch2a_scale)
modmap[#modmap+1] = {scale4b11_branch2a_scale, scale4b11_branch2a}

res4b11_branch2a_relu = nn.ReLU(true)(scale4b11_branch2a)
modmap[#modmap+1] = {res4b11_branch2a_relu}

res4b11_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b11_branch2a_relu)
modmap[#modmap+1] = {res4b11_branch2b}

bn4b11_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b11_branch2b)
modmap[#modmap+1] = {bn4b11_branch2b}

scale4b11_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b11_branch2b)
scale4b11_branch2b = nn.Add(1)(scale4b11_branch2b_scale)
modmap[#modmap+1] = {scale4b11_branch2b_scale, scale4b11_branch2b}

res4b11_branch2b_relu = nn.ReLU(true)(scale4b11_branch2b)
modmap[#modmap+1] = {res4b11_branch2b_relu}

res4b11_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b11_branch2b_relu)
modmap[#modmap+1] = {res4b11_branch2c}

bn4b11_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b11_branch2c)
modmap[#modmap+1] = {bn4b11_branch2c}

scale4b11_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b11_branch2c)
scale4b11_branch2c = nn.Add(1)(scale4b11_branch2c_scale)
modmap[#modmap+1] = {scale4b11_branch2c_scale, scale4b11_branch2c}

res4b11 = nn.CAddTable()({res4b10_relu, scale4b11_branch2c})
modmap[#modmap+1] = {res4b11}

res4b11_relu = nn.ReLU(true)(res4b11)
modmap[#modmap+1] = {res4b11_relu}

res4b12_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b11_relu)
modmap[#modmap+1] = {res4b12_branch2a}

bn4b12_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b12_branch2a)
modmap[#modmap+1] = {bn4b12_branch2a}

scale4b12_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b12_branch2a)
scale4b12_branch2a = nn.Add(1)(scale4b12_branch2a_scale)
modmap[#modmap+1] = {scale4b12_branch2a_scale, scale4b12_branch2a}

res4b12_branch2a_relu = nn.ReLU(true)(scale4b12_branch2a)
modmap[#modmap+1] = {res4b12_branch2a_relu}

res4b12_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b12_branch2a_relu)
modmap[#modmap+1] = {res4b12_branch2b}

bn4b12_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b12_branch2b)
modmap[#modmap+1] = {bn4b12_branch2b}

scale4b12_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b12_branch2b)
scale4b12_branch2b = nn.Add(1)(scale4b12_branch2b_scale)
modmap[#modmap+1] = {scale4b12_branch2b_scale, scale4b12_branch2b}

res4b12_branch2b_relu = nn.ReLU(true)(scale4b12_branch2b)
modmap[#modmap+1] = {res4b12_branch2b_relu}

res4b12_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b12_branch2b_relu)
modmap[#modmap+1] = {res4b12_branch2c}

bn4b12_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b12_branch2c)
modmap[#modmap+1] = {bn4b12_branch2c}

scale4b12_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b12_branch2c)
scale4b12_branch2c = nn.Add(1)(scale4b12_branch2c_scale)
modmap[#modmap+1] = {scale4b12_branch2c_scale, scale4b12_branch2c}

res4b12 = nn.CAddTable()({res4b11_relu, scale4b12_branch2c})
modmap[#modmap+1] = {res4b12}

res4b12_relu = nn.ReLU(true)(res4b12)
modmap[#modmap+1] = {res4b12_relu}

res4b13_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b12_relu)
modmap[#modmap+1] = {res4b13_branch2a}

bn4b13_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b13_branch2a)
modmap[#modmap+1] = {bn4b13_branch2a}

scale4b13_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b13_branch2a)
scale4b13_branch2a = nn.Add(1)(scale4b13_branch2a_scale)
modmap[#modmap+1] = {scale4b13_branch2a_scale, scale4b13_branch2a}

res4b13_branch2a_relu = nn.ReLU(true)(scale4b13_branch2a)
modmap[#modmap+1] = {res4b13_branch2a_relu}

res4b13_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b13_branch2a_relu)
modmap[#modmap+1] = {res4b13_branch2b}

bn4b13_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b13_branch2b)
modmap[#modmap+1] = {bn4b13_branch2b}

scale4b13_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b13_branch2b)
scale4b13_branch2b = nn.Add(1)(scale4b13_branch2b_scale)
modmap[#modmap+1] = {scale4b13_branch2b_scale, scale4b13_branch2b}

res4b13_branch2b_relu = nn.ReLU(true)(scale4b13_branch2b)
modmap[#modmap+1] = {res4b13_branch2b_relu}

res4b13_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b13_branch2b_relu)
modmap[#modmap+1] = {res4b13_branch2c}

bn4b13_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b13_branch2c)
modmap[#modmap+1] = {bn4b13_branch2c}

scale4b13_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b13_branch2c)
scale4b13_branch2c = nn.Add(1)(scale4b13_branch2c_scale)
modmap[#modmap+1] = {scale4b13_branch2c_scale, scale4b13_branch2c}

res4b13 = nn.CAddTable()({res4b12_relu, scale4b13_branch2c})
modmap[#modmap+1] = {res4b13}

res4b13_relu = nn.ReLU(true)(res4b13)
modmap[#modmap+1] = {res4b13_relu}

res4b14_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b13_relu)
modmap[#modmap+1] = {res4b14_branch2a}

bn4b14_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b14_branch2a)
modmap[#modmap+1] = {bn4b14_branch2a}

scale4b14_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b14_branch2a)
scale4b14_branch2a = nn.Add(1)(scale4b14_branch2a_scale)
modmap[#modmap+1] = {scale4b14_branch2a_scale, scale4b14_branch2a}

res4b14_branch2a_relu = nn.ReLU(true)(scale4b14_branch2a)
modmap[#modmap+1] = {res4b14_branch2a_relu}

res4b14_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b14_branch2a_relu)
modmap[#modmap+1] = {res4b14_branch2b}

bn4b14_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b14_branch2b)
modmap[#modmap+1] = {bn4b14_branch2b}

scale4b14_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b14_branch2b)
scale4b14_branch2b = nn.Add(1)(scale4b14_branch2b_scale)
modmap[#modmap+1] = {scale4b14_branch2b_scale, scale4b14_branch2b}

res4b14_branch2b_relu = nn.ReLU(true)(scale4b14_branch2b)
modmap[#modmap+1] = {res4b14_branch2b_relu}

res4b14_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b14_branch2b_relu)
modmap[#modmap+1] = {res4b14_branch2c}

bn4b14_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b14_branch2c)
modmap[#modmap+1] = {bn4b14_branch2c}

scale4b14_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b14_branch2c)
scale4b14_branch2c = nn.Add(1)(scale4b14_branch2c_scale)
modmap[#modmap+1] = {scale4b14_branch2c_scale, scale4b14_branch2c}

res4b14 = nn.CAddTable()({res4b13_relu, scale4b14_branch2c})
modmap[#modmap+1] = {res4b14}

res4b14_relu = nn.ReLU(true)(res4b14)
modmap[#modmap+1] = {res4b14_relu}

res4b15_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b14_relu)
modmap[#modmap+1] = {res4b15_branch2a}

bn4b15_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b15_branch2a)
modmap[#modmap+1] = {bn4b15_branch2a}

scale4b15_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b15_branch2a)
scale4b15_branch2a = nn.Add(1)(scale4b15_branch2a_scale)
modmap[#modmap+1] = {scale4b15_branch2a_scale, scale4b15_branch2a}

res4b15_branch2a_relu = nn.ReLU(true)(scale4b15_branch2a)
modmap[#modmap+1] = {res4b15_branch2a_relu}

res4b15_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b15_branch2a_relu)
modmap[#modmap+1] = {res4b15_branch2b}

bn4b15_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b15_branch2b)
modmap[#modmap+1] = {bn4b15_branch2b}

scale4b15_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b15_branch2b)
scale4b15_branch2b = nn.Add(1)(scale4b15_branch2b_scale)
modmap[#modmap+1] = {scale4b15_branch2b_scale, scale4b15_branch2b}

res4b15_branch2b_relu = nn.ReLU(true)(scale4b15_branch2b)
modmap[#modmap+1] = {res4b15_branch2b_relu}

res4b15_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b15_branch2b_relu)
modmap[#modmap+1] = {res4b15_branch2c}

bn4b15_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b15_branch2c)
modmap[#modmap+1] = {bn4b15_branch2c}

scale4b15_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b15_branch2c)
scale4b15_branch2c = nn.Add(1)(scale4b15_branch2c_scale)
modmap[#modmap+1] = {scale4b15_branch2c_scale, scale4b15_branch2c}

res4b15 = nn.CAddTable()({res4b14_relu, scale4b15_branch2c})
modmap[#modmap+1] = {res4b15}

res4b15_relu = nn.ReLU(true)(res4b15)
modmap[#modmap+1] = {res4b15_relu}

res4b16_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b15_relu)
modmap[#modmap+1] = {res4b16_branch2a}

bn4b16_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b16_branch2a)
modmap[#modmap+1] = {bn4b16_branch2a}

scale4b16_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b16_branch2a)
scale4b16_branch2a = nn.Add(1)(scale4b16_branch2a_scale)
modmap[#modmap+1] = {scale4b16_branch2a_scale, scale4b16_branch2a}

res4b16_branch2a_relu = nn.ReLU(true)(scale4b16_branch2a)
modmap[#modmap+1] = {res4b16_branch2a_relu}

res4b16_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b16_branch2a_relu)
modmap[#modmap+1] = {res4b16_branch2b}

bn4b16_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b16_branch2b)
modmap[#modmap+1] = {bn4b16_branch2b}

scale4b16_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b16_branch2b)
scale4b16_branch2b = nn.Add(1)(scale4b16_branch2b_scale)
modmap[#modmap+1] = {scale4b16_branch2b_scale, scale4b16_branch2b}

res4b16_branch2b_relu = nn.ReLU(true)(scale4b16_branch2b)
modmap[#modmap+1] = {res4b16_branch2b_relu}

res4b16_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b16_branch2b_relu)
modmap[#modmap+1] = {res4b16_branch2c}

bn4b16_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b16_branch2c)
modmap[#modmap+1] = {bn4b16_branch2c}

scale4b16_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b16_branch2c)
scale4b16_branch2c = nn.Add(1)(scale4b16_branch2c_scale)
modmap[#modmap+1] = {scale4b16_branch2c_scale, scale4b16_branch2c}

res4b16 = nn.CAddTable()({res4b15_relu, scale4b16_branch2c})
modmap[#modmap+1] = {res4b16}

res4b16_relu = nn.ReLU(true)(res4b16)
modmap[#modmap+1] = {res4b16_relu}

res4b17_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b16_relu)
modmap[#modmap+1] = {res4b17_branch2a}

bn4b17_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b17_branch2a)
modmap[#modmap+1] = {bn4b17_branch2a}

scale4b17_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b17_branch2a)
scale4b17_branch2a = nn.Add(1)(scale4b17_branch2a_scale)
modmap[#modmap+1] = {scale4b17_branch2a_scale, scale4b17_branch2a}

res4b17_branch2a_relu = nn.ReLU(true)(scale4b17_branch2a)
modmap[#modmap+1] = {res4b17_branch2a_relu}

res4b17_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b17_branch2a_relu)
modmap[#modmap+1] = {res4b17_branch2b}

bn4b17_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b17_branch2b)
modmap[#modmap+1] = {bn4b17_branch2b}

scale4b17_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b17_branch2b)
scale4b17_branch2b = nn.Add(1)(scale4b17_branch2b_scale)
modmap[#modmap+1] = {scale4b17_branch2b_scale, scale4b17_branch2b}

res4b17_branch2b_relu = nn.ReLU(true)(scale4b17_branch2b)
modmap[#modmap+1] = {res4b17_branch2b_relu}

res4b17_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b17_branch2b_relu)
modmap[#modmap+1] = {res4b17_branch2c}

bn4b17_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b17_branch2c)
modmap[#modmap+1] = {bn4b17_branch2c}

scale4b17_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b17_branch2c)
scale4b17_branch2c = nn.Add(1)(scale4b17_branch2c_scale)
modmap[#modmap+1] = {scale4b17_branch2c_scale, scale4b17_branch2c}

res4b17 = nn.CAddTable()({res4b16_relu, scale4b17_branch2c})
modmap[#modmap+1] = {res4b17}

res4b17_relu = nn.ReLU(true)(res4b17)
modmap[#modmap+1] = {res4b17_relu}

res4b18_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b17_relu)
modmap[#modmap+1] = {res4b18_branch2a}

bn4b18_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b18_branch2a)
modmap[#modmap+1] = {bn4b18_branch2a}

scale4b18_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b18_branch2a)
scale4b18_branch2a = nn.Add(1)(scale4b18_branch2a_scale)
modmap[#modmap+1] = {scale4b18_branch2a_scale, scale4b18_branch2a}

res4b18_branch2a_relu = nn.ReLU(true)(scale4b18_branch2a)
modmap[#modmap+1] = {res4b18_branch2a_relu}

res4b18_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b18_branch2a_relu)
modmap[#modmap+1] = {res4b18_branch2b}

bn4b18_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b18_branch2b)
modmap[#modmap+1] = {bn4b18_branch2b}

scale4b18_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b18_branch2b)
scale4b18_branch2b = nn.Add(1)(scale4b18_branch2b_scale)
modmap[#modmap+1] = {scale4b18_branch2b_scale, scale4b18_branch2b}

res4b18_branch2b_relu = nn.ReLU(true)(scale4b18_branch2b)
modmap[#modmap+1] = {res4b18_branch2b_relu}

res4b18_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b18_branch2b_relu)
modmap[#modmap+1] = {res4b18_branch2c}

bn4b18_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b18_branch2c)
modmap[#modmap+1] = {bn4b18_branch2c}

scale4b18_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b18_branch2c)
scale4b18_branch2c = nn.Add(1)(scale4b18_branch2c_scale)
modmap[#modmap+1] = {scale4b18_branch2c_scale, scale4b18_branch2c}

res4b18 = nn.CAddTable()({res4b17_relu, scale4b18_branch2c})
modmap[#modmap+1] = {res4b18}

res4b18_relu = nn.ReLU(true)(res4b18)
modmap[#modmap+1] = {res4b18_relu}

res4b19_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b18_relu)
modmap[#modmap+1] = {res4b19_branch2a}

bn4b19_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b19_branch2a)
modmap[#modmap+1] = {bn4b19_branch2a}

scale4b19_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b19_branch2a)
scale4b19_branch2a = nn.Add(1)(scale4b19_branch2a_scale)
modmap[#modmap+1] = {scale4b19_branch2a_scale, scale4b19_branch2a}

res4b19_branch2a_relu = nn.ReLU(true)(scale4b19_branch2a)
modmap[#modmap+1] = {res4b19_branch2a_relu}

res4b19_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b19_branch2a_relu)
modmap[#modmap+1] = {res4b19_branch2b}

bn4b19_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b19_branch2b)
modmap[#modmap+1] = {bn4b19_branch2b}

scale4b19_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b19_branch2b)
scale4b19_branch2b = nn.Add(1)(scale4b19_branch2b_scale)
modmap[#modmap+1] = {scale4b19_branch2b_scale, scale4b19_branch2b}

res4b19_branch2b_relu = nn.ReLU(true)(scale4b19_branch2b)
modmap[#modmap+1] = {res4b19_branch2b_relu}

res4b19_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b19_branch2b_relu)
modmap[#modmap+1] = {res4b19_branch2c}

bn4b19_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b19_branch2c)
modmap[#modmap+1] = {bn4b19_branch2c}

scale4b19_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b19_branch2c)
scale4b19_branch2c = nn.Add(1)(scale4b19_branch2c_scale)
modmap[#modmap+1] = {scale4b19_branch2c_scale, scale4b19_branch2c}

res4b19 = nn.CAddTable()({res4b18_relu, scale4b19_branch2c})
modmap[#modmap+1] = {res4b19}

res4b19_relu = nn.ReLU(true)(res4b19)
modmap[#modmap+1] = {res4b19_relu}

res4b20_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b19_relu)
modmap[#modmap+1] = {res4b20_branch2a}

bn4b20_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b20_branch2a)
modmap[#modmap+1] = {bn4b20_branch2a}

scale4b20_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b20_branch2a)
scale4b20_branch2a = nn.Add(1)(scale4b20_branch2a_scale)
modmap[#modmap+1] = {scale4b20_branch2a_scale, scale4b20_branch2a}

res4b20_branch2a_relu = nn.ReLU(true)(scale4b20_branch2a)
modmap[#modmap+1] = {res4b20_branch2a_relu}

res4b20_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b20_branch2a_relu)
modmap[#modmap+1] = {res4b20_branch2b}

bn4b20_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b20_branch2b)
modmap[#modmap+1] = {bn4b20_branch2b}

scale4b20_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b20_branch2b)
scale4b20_branch2b = nn.Add(1)(scale4b20_branch2b_scale)
modmap[#modmap+1] = {scale4b20_branch2b_scale, scale4b20_branch2b}

res4b20_branch2b_relu = nn.ReLU(true)(scale4b20_branch2b)
modmap[#modmap+1] = {res4b20_branch2b_relu}

res4b20_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b20_branch2b_relu)
modmap[#modmap+1] = {res4b20_branch2c}

bn4b20_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b20_branch2c)
modmap[#modmap+1] = {bn4b20_branch2c}

scale4b20_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b20_branch2c)
scale4b20_branch2c = nn.Add(1)(scale4b20_branch2c_scale)
modmap[#modmap+1] = {scale4b20_branch2c_scale, scale4b20_branch2c}

res4b20 = nn.CAddTable()({res4b19_relu, scale4b20_branch2c})
modmap[#modmap+1] = {res4b20}

res4b20_relu = nn.ReLU(true)(res4b20)
modmap[#modmap+1] = {res4b20_relu}

res4b21_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b20_relu)
modmap[#modmap+1] = {res4b21_branch2a}

bn4b21_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b21_branch2a)
modmap[#modmap+1] = {bn4b21_branch2a}

scale4b21_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b21_branch2a)
scale4b21_branch2a = nn.Add(1)(scale4b21_branch2a_scale)
modmap[#modmap+1] = {scale4b21_branch2a_scale, scale4b21_branch2a}

res4b21_branch2a_relu = nn.ReLU(true)(scale4b21_branch2a)
modmap[#modmap+1] = {res4b21_branch2a_relu}

res4b21_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b21_branch2a_relu)
modmap[#modmap+1] = {res4b21_branch2b}

bn4b21_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b21_branch2b)
modmap[#modmap+1] = {bn4b21_branch2b}

scale4b21_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b21_branch2b)
scale4b21_branch2b = nn.Add(1)(scale4b21_branch2b_scale)
modmap[#modmap+1] = {scale4b21_branch2b_scale, scale4b21_branch2b}

res4b21_branch2b_relu = nn.ReLU(true)(scale4b21_branch2b)
modmap[#modmap+1] = {res4b21_branch2b_relu}

res4b21_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b21_branch2b_relu)
modmap[#modmap+1] = {res4b21_branch2c}

bn4b21_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b21_branch2c)
modmap[#modmap+1] = {bn4b21_branch2c}

scale4b21_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b21_branch2c)
scale4b21_branch2c = nn.Add(1)(scale4b21_branch2c_scale)
modmap[#modmap+1] = {scale4b21_branch2c_scale, scale4b21_branch2c}

res4b21 = nn.CAddTable()({res4b20_relu, scale4b21_branch2c})
modmap[#modmap+1] = {res4b21}

res4b21_relu = nn.ReLU(true)(res4b21)
modmap[#modmap+1] = {res4b21_relu}

res4b22_branch2a = nn.SpatialConvolution(1024, 256, 1, 1, 1, 1, 0, 0)(res4b21_relu)
modmap[#modmap+1] = {res4b22_branch2a}

bn4b22_branch2a = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b22_branch2a)
modmap[#modmap+1] = {bn4b22_branch2a}

scale4b22_branch2a_scale = nn.CMul(1, 256, 1, 1)(bn4b22_branch2a)
scale4b22_branch2a = nn.Add(1)(scale4b22_branch2a_scale)
modmap[#modmap+1] = {scale4b22_branch2a_scale, scale4b22_branch2a}

res4b22_branch2a_relu = nn.ReLU(true)(scale4b22_branch2a)
modmap[#modmap+1] = {res4b22_branch2a_relu}

res4b22_branch2b = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(res4b22_branch2a_relu)
modmap[#modmap+1] = {res4b22_branch2b}

bn4b22_branch2b = nn.SpatialBatchNormalization(256, 1e-05, 0.999)(res4b22_branch2b)
modmap[#modmap+1] = {bn4b22_branch2b}

scale4b22_branch2b_scale = nn.CMul(1, 256, 1, 1)(bn4b22_branch2b)
scale4b22_branch2b = nn.Add(1)(scale4b22_branch2b_scale)
modmap[#modmap+1] = {scale4b22_branch2b_scale, scale4b22_branch2b}

res4b22_branch2b_relu = nn.ReLU(true)(scale4b22_branch2b)
modmap[#modmap+1] = {res4b22_branch2b_relu}

res4b22_branch2c = nn.SpatialConvolution(256, 1024, 1, 1, 1, 1, 0, 0)(res4b22_branch2b_relu)
modmap[#modmap+1] = {res4b22_branch2c}

bn4b22_branch2c = nn.SpatialBatchNormalization(1024, 1e-05, 0.999)(res4b22_branch2c)
modmap[#modmap+1] = {bn4b22_branch2c}

scale4b22_branch2c_scale = nn.CMul(1, 1024, 1, 1)(bn4b22_branch2c)
scale4b22_branch2c = nn.Add(1)(scale4b22_branch2c_scale)
modmap[#modmap+1] = {scale4b22_branch2c_scale, scale4b22_branch2c}

res4b22 = nn.CAddTable()({res4b21_relu, scale4b22_branch2c})
modmap[#modmap+1] = {res4b22}

res4b22_relu = nn.ReLU(true)(res4b22)
modmap[#modmap+1] = {res4b22_relu}

res5a_branch1 = nn.SpatialConvolution(1024, 2048, 1, 1, 2, 2, 0, 0)(res4b22_relu)
modmap[#modmap+1] = {res5a_branch1}

bn5a_branch1 = nn.SpatialBatchNormalization(2048, 1e-05, 0.999)(res5a_branch1)
modmap[#modmap+1] = {bn5a_branch1}

scale5a_branch1_scale = nn.CMul(1, 2048, 1, 1)(bn5a_branch1)
scale5a_branch1 = nn.Add(1)(scale5a_branch1_scale)
modmap[#modmap+1] = {scale5a_branch1_scale, scale5a_branch1}

res5a_branch2a = nn.SpatialConvolution(1024, 512, 1, 1, 2, 2, 0, 0)(res4b22_relu)
modmap[#modmap+1] = {res5a_branch2a}

bn5a_branch2a = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res5a_branch2a)
modmap[#modmap+1] = {bn5a_branch2a}

scale5a_branch2a_scale = nn.CMul(1, 512, 1, 1)(bn5a_branch2a)
scale5a_branch2a = nn.Add(1)(scale5a_branch2a_scale)
modmap[#modmap+1] = {scale5a_branch2a_scale, scale5a_branch2a}

res5a_branch2a_relu = nn.ReLU(true)(scale5a_branch2a)
modmap[#modmap+1] = {res5a_branch2a_relu}

res5a_branch2b = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(res5a_branch2a_relu)
modmap[#modmap+1] = {res5a_branch2b}

bn5a_branch2b = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res5a_branch2b)
modmap[#modmap+1] = {bn5a_branch2b}

scale5a_branch2b_scale = nn.CMul(1, 512, 1, 1)(bn5a_branch2b)
scale5a_branch2b = nn.Add(1)(scale5a_branch2b_scale)
modmap[#modmap+1] = {scale5a_branch2b_scale, scale5a_branch2b}

res5a_branch2b_relu = nn.ReLU(true)(scale5a_branch2b)
modmap[#modmap+1] = {res5a_branch2b_relu}

res5a_branch2c = nn.SpatialConvolution(512, 2048, 1, 1, 1, 1, 0, 0)(res5a_branch2b_relu)
modmap[#modmap+1] = {res5a_branch2c}

bn5a_branch2c = nn.SpatialBatchNormalization(2048, 1e-05, 0.999)(res5a_branch2c)
modmap[#modmap+1] = {bn5a_branch2c}

scale5a_branch2c_scale = nn.CMul(1, 2048, 1, 1)(bn5a_branch2c)
scale5a_branch2c = nn.Add(1)(scale5a_branch2c_scale)
modmap[#modmap+1] = {scale5a_branch2c_scale, scale5a_branch2c}

res5a = nn.CAddTable()({scale5a_branch1, scale5a_branch2c})
modmap[#modmap+1] = {res5a}

res5a_relu = nn.ReLU(true)(res5a)
modmap[#modmap+1] = {res5a_relu}

res5b_branch2a = nn.SpatialConvolution(2048, 512, 1, 1, 1, 1, 0, 0)(res5a_relu)
modmap[#modmap+1] = {res5b_branch2a}

bn5b_branch2a = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res5b_branch2a)
modmap[#modmap+1] = {bn5b_branch2a}

scale5b_branch2a_scale = nn.CMul(1, 512, 1, 1)(bn5b_branch2a)
scale5b_branch2a = nn.Add(1)(scale5b_branch2a_scale)
modmap[#modmap+1] = {scale5b_branch2a_scale, scale5b_branch2a}

res5b_branch2a_relu = nn.ReLU(true)(scale5b_branch2a)
modmap[#modmap+1] = {res5b_branch2a_relu}

res5b_branch2b = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(res5b_branch2a_relu)
modmap[#modmap+1] = {res5b_branch2b}

bn5b_branch2b = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res5b_branch2b)
modmap[#modmap+1] = {bn5b_branch2b}

scale5b_branch2b_scale = nn.CMul(1, 512, 1, 1)(bn5b_branch2b)
scale5b_branch2b = nn.Add(1)(scale5b_branch2b_scale)
modmap[#modmap+1] = {scale5b_branch2b_scale, scale5b_branch2b}

res5b_branch2b_relu = nn.ReLU(true)(scale5b_branch2b)
modmap[#modmap+1] = {res5b_branch2b_relu}

res5b_branch2c = nn.SpatialConvolution(512, 2048, 1, 1, 1, 1, 0, 0)(res5b_branch2b_relu)
modmap[#modmap+1] = {res5b_branch2c}

bn5b_branch2c = nn.SpatialBatchNormalization(2048, 1e-05, 0.999)(res5b_branch2c)
modmap[#modmap+1] = {bn5b_branch2c}

scale5b_branch2c_scale = nn.CMul(1, 2048, 1, 1)(bn5b_branch2c)
scale5b_branch2c = nn.Add(1)(scale5b_branch2c_scale)
modmap[#modmap+1] = {scale5b_branch2c_scale, scale5b_branch2c}

res5b = nn.CAddTable()({res5a_relu, scale5b_branch2c})
modmap[#modmap+1] = {res5b}

res5b_relu = nn.ReLU(true)(res5b)
modmap[#modmap+1] = {res5b_relu}

res5c_branch2a = nn.SpatialConvolution(2048, 512, 1, 1, 1, 1, 0, 0)(res5b_relu)
modmap[#modmap+1] = {res5c_branch2a}

bn5c_branch2a = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res5c_branch2a)
modmap[#modmap+1] = {bn5c_branch2a}

scale5c_branch2a_scale = nn.CMul(1, 512, 1, 1)(bn5c_branch2a)
scale5c_branch2a = nn.Add(1)(scale5c_branch2a_scale)
modmap[#modmap+1] = {scale5c_branch2a_scale, scale5c_branch2a}

res5c_branch2a_relu = nn.ReLU(true)(scale5c_branch2a)
modmap[#modmap+1] = {res5c_branch2a_relu}

res5c_branch2b = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(res5c_branch2a_relu)
modmap[#modmap+1] = {res5c_branch2b}

bn5c_branch2b = nn.SpatialBatchNormalization(512, 1e-05, 0.999)(res5c_branch2b)
modmap[#modmap+1] = {bn5c_branch2b}

scale5c_branch2b_scale = nn.CMul(1, 512, 1, 1)(bn5c_branch2b)
scale5c_branch2b = nn.Add(1)(scale5c_branch2b_scale)
modmap[#modmap+1] = {scale5c_branch2b_scale, scale5c_branch2b}

res5c_branch2b_relu = nn.ReLU(true)(scale5c_branch2b)
modmap[#modmap+1] = {res5c_branch2b_relu}

res5c_branch2c = nn.SpatialConvolution(512, 2048, 1, 1, 1, 1, 0, 0)(res5c_branch2b_relu)
modmap[#modmap+1] = {res5c_branch2c}

bn5c_branch2c = nn.SpatialBatchNormalization(2048, 1e-05, 0.999)(res5c_branch2c)
modmap[#modmap+1] = {bn5c_branch2c}

scale5c_branch2c_scale = nn.CMul(1, 2048, 1, 1)(bn5c_branch2c)
scale5c_branch2c = nn.Add(1)(scale5c_branch2c_scale)
modmap[#modmap+1] = {scale5c_branch2c_scale, scale5c_branch2c}

res5c = nn.CAddTable()({res5b_relu, scale5c_branch2c})
modmap[#modmap+1] = {res5c}

res5c_relu = nn.ReLU(true)(res5c)
modmap[#modmap+1] = {res5c_relu}

pool5 = nn.SpatialAveragePooling(7, 7, 1, 1, 0, 0):ceil()(res5c_relu)
modmap[#modmap+1] = {pool5}

collapse = nn.View(-1):setNumInputDims(3)(pool5)
fc1000 = nn.Linear(2048, 1000)(collapse)
modmap[#modmap+1] = {collapse, fc1000}

prob = nn.SoftMax()(fc1000)
modmap[#modmap+1] = {prob}

model = nn.gModule({data}, {prob})

return model, modmap
