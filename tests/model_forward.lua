--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
require 'nn'
require 'cudnn'
require 'nngraph'


paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)

outputDim={}
table.insert(outputDim,{11})

local models_list = paths.dofile('../models/init.lua')
model = models_list['hg-generic-concat']()

local input = torch.CudaTensor(2,3,224,224):uniform()


cudnn.convert(model, cudnn):cuda()
res = model:forward(input)

print(#res)
