require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'nngraph'

cutorch.setDevice(2)
--[[
paths.dofile('Inception.lua')

local resid = InceptionResidual(256,256):cuda()
print(resid)
-]]

paths.dofile('WideResidual.lua')
local resid = WideResidual(256,256,1,1):cuda()
print(resid)

local input = torch.Tensor(4,256,32,32):cuda()

-- 
local res = resid:forward(input)

print(#res)

