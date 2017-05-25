--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
require 'nn'
require 'cudnn'
require 'nngraph'

torch.setdefaulttensortype('torch.FloatTensor')

local i1 = nn.Identity()()
local i2 = nn.Identity()()

local conv1 = nn.SpatialConvolutionMM(10,20,3,3,1,1)(i1)
local conv2 = nn.SpatialConvolutionMM(10,20,3,3,1,1)(i2)
local join = nn.JoinTable(2)({conv1,conv2})

local model = nn.gModule({i1,i2},{join})

graph.dot(model.fg, 'pose network')

local input = {torch.Tensor(2,10,20,20):uniform(), torch.Tensor(2,10,20,20):uniform()}

local out = model:forward(input)

print(#out)