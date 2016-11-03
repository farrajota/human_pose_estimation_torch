--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'


paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)

paths.dofile('../dataset.lua')
paths.dofile('../data.lua')

dataset = loadDataset() -- load dataset train+val+test sets
a,b = loadData(dataset.train, 19)

print('Input size')
print(#a)
print('heatmap size')
print(#b)
