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
paths.dofile('../util/eval.lua')
paths.dofile('../data.lua')
paths.dofile('../sample_batch.lua')

paths.dofile('../util/draw.lua')
paths.dofile('../util/eval.lua')
paths.dofile('../util/img.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'lsp+mpii'
opt.rotRate = .2
niters = 10000
mode = 'train'
plot_results = false
opt.subpixel_precision = true

local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]

for i=1, niters do
    print(('Iter %d/%d'):format(i, niters))
    if i==6 then
        a=1  -- stop debugger here
    end
    local input, label, keypoints = getSampleBatch(loader, opt.batchSize)

  
    c = getPreds(label)

    if plot_results then
        a = {}
        for i=1, input:size(1) do
            local hm = label[i]
            local im = input[i]
            table.insert(a, drawImgHeatmapSingle(im, hm:sum(1):squeeze()))
        end
        disp.image(input)
        disp.image(a)
    end
end

print('Data fetching successfully finished.')
