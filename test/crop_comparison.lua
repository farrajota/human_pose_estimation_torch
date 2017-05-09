--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
disp = require 'display'

torch.manualSeed(4)

paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'lsp'
opt.rotRate=0

paths.dofile('../dataset.lua')
paths.dofile('../data.lua')

dataset = loadDataset() -- load dataset train+val+test sets

mode = 'train'
idx = 3500

local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

local r = 0 -- set rotation to 0

-- Load image + keypoints + other data
local img, keypoints, c, s, nJoints = loadImgKeypointsFn_(dataset[mode], idx)  
local s_ = s

-- Do rotation + scaling
if mode == 'train' then
    -- Scale and rotation augmentation
    --s = s * torch.uniform(1-opt.scale, 1+opt.scale)
    s = s * (2 ^ rnd(opt.scale))
    r = rnd(opt.rotate)
    --r = torch.uniform(-opt.rotate, opt.rotate)
    if torch.uniform() <= opt.rotRate then r = 0 end
end

----------------------------------------------------------------------------
-- my crop -----------------------------------------------------------------
----------------------------------------------------------------------------

-- Crop image + craft heatmap
local kps1 = {}
local img_transf1 = mycrop(img, c, s, r, opt.inputRes)
local heatmap1 = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
for i = 1,nJoints do
    if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
        local new_kp = mytransform(keypoints[i], c, s, r, opt.outputRes)
        drawGaussian(heatmap1[i], new_kp:int(), opt.hmGauss)
        table.insert(kps1, new_kp:totable())
    end
end

disp.image(img_transf1,{title='my crop'})

----------------------------------------------------------------------------
-- crop2 ------------------------------------------------------------------
----------------------------------------------------------------------------

 -- Crop image + craft heatmap
local kps2 = {}
local img_transf2 = crop2(img, c, s, r, opt.inputRes)
local heatmap2 = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
for i = 1,nJoints do
    if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
        local new_kp = transform(torch.add(keypoints[i],1), c, s, r, opt.outputRes)
        drawGaussian(heatmap2[i], new_kp, opt.hmGauss)
        table.insert(kps2, new_kp:totable())
    end
end

disp.image(img_transf2,{title='crop2'})

disp.image(torch.csub(img_transf1:float(),img_transf2:float()),{title='difference'})
disp.image(torch.csub(heatmap1:float(),heatmap2:float()),{title='difference heatmaps'})

print('Keypoints my:')
print(torch.FloatTensor(kps1))

print('Keypoints crop2:')
print(torch.FloatTensor(kps2))