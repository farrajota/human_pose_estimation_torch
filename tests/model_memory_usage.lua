require 'paths'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
optnet = require 'optnet'


torch.setdefaulttensortype('torch.FloatTensor')

opt = {}
opt.dropout = 0
opt.spatialdropout = 0.2
opt.nonlinearity = 'relu'
opt.nFeats = 256
opt.nStack = 8
opt.inputRes = 256
opt.outputRes = 256
outputDim = {{16}}



local Residual, relu = paths.dofile('../models/layers/ResidualSML.lua')


local function model_maxpool_upsampling()
    local input = nn.Identity()()

    -- Initial processing of the image
    local cnv1 = nn.Sequential()
        :add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))     -- 128
        :add(nn.SpatialBatchNormalization(64))
        :add(relu())(input)
    local r1 = Residual(64,128)(cnv1)                -- 64
    local pool1 = nn.SpatialMaxPooling(2,2,2,2)(r1)
    local r2 = Residual(128,128)(pool1)
    local pool2 = nn.SpatialMaxPooling(2,2,2,2)(r2)
    local r3 = Residual(128,128)(pool2)
    local up1 = nn.SpatialUpSamplingNearest(2)(r3)
    local r4 = Residual(128,256)(up1)
    local up2 = nn.SpatialUpSamplingNearest(2)(r4)

    return nn.gModule({input}, {up2})
end

local function model_fullconv()
    local input = nn.Identity()()

    -- Initial processing of the image
    local cnv1 = nn.Sequential()
        :add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))     -- 128
        :add(nn.SpatialBatchNormalization(64))
        :add(relu())(input)
    local r1 = Residual(64,128,  'D')(cnv1)                -- 64
    local r2 = Residual(128,128, 'D')(r1)
    local r3 = Residual(128,128, 'U')(r2)
    local r4 = Residual(128,256, 'U')(r3)

    return nn.gModule({input}, {r4})
end


print('\n****************************************')
print('Testing models memory usage: ')
print('   - Model1 (maxpool + upsampling layers)')
print('   - Model2 (full convolutions)')
print('****************************************\n')

--[[ Setup Input tensor ]]--
local input = torch.Tensor(4,3,256,256):uniform()

--[[ Test model with maxpool+upsampling layers]]
local mem1, mem2
do
    local model = model_maxpool_upsampling()
    local res = model:forward(input)
    mem1 = optnet.countUsedMemory(model)
    print('Model1 memory usage: ' .. mem1.total_size/1024/1024)
end

--[[ Test model without maxpool+upsampling layers]]
do
    local model = model_fullconv()
    local res = model:forward(input)
    mem2 = optnet.countUsedMemory(model)
    print('Model2 memory usage: ' .. mem2.total_size/1024/1024)
end


print('Memory savings: ' .. (mem2.total_size/mem1.total_size))