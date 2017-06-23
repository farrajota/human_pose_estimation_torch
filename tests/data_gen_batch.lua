--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
disp = require 'display'

torch.manualSeed(4)
torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')
paths.dofile('../data.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'mpii+lsp'
opt.rotRate=0.2
niters = 100
mode = 'train'

local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]

for i=1, niters do
    print(('Iter %d/%d'):format(i, niters))
    if i==24 then
      aqui=1
    end
    local input, label = getSampleBatch(loader, opt.batchSize)
end

print('Data fetching successfully finished.')
