--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
disp = require 'display'

paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'mpii'

paths.dofile('../dataset.lua')
paths.dofile('../data.lua')

dataset = loadDataset() -- load dataset train+val+test sets
--a,b = loadDataBenchmark(1,dataset.val, 'val')
mode = 'train'
--a,b = loadData(1,dataset[mode], mode)
N = 9;
local img = torch.FloatTensor(N,3,256,256);
for i=1, N do
  img[i], _ = loadData(3500,dataset[mode], mode)
end



disp.image(img)

--print('Input size')
--print(#a)
--print('heatmap size')
--print(#b)

