--[[
    Data sampling functions.
]]


require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/utils.lua')

local ffi = require 'ffi'
local t = paths.dofile('transforms.lua')


-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end


-------------------------------------------------------------------------------
--Image + keypoint loading functions
-------------------------------------------------------------------------------

local function MPIILoadImgKeypointsFn(data, idx)
    local object = data.object[idx]
    local filename = ffi.string(data.filename[object[1]]:data())
    local keypoints = data.keypoint[object[3]]
    local center = data.objpos[object[5]]
    local scale = data.scale[object[4]]
    local nJoints = keypoints:size(1)/3
    
    -- Small adjustment so cropping is less likely to take feet out
    center[2] = center[2] + 15 * scale
    scale = scale * 1.25
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints:view(nJoints, 3), center, scale, nJoints
end

local function FLICLoadImgKeypointsFn(idx)
    -- initializations
    local object = data.object[idx]
    local filename = ffi.string(data.filename[object[1]]:data())
    local keypoints = data.keypoints[object[3]]
    local torso = data.torso[object[4]]
    local center = torch.FloatTensor({(torso[1]+torso[3])/2, (torso[2]+torso[4])/2})
    local scale = 2.2
    
    
end

local function LSPLoadImgKeypointsFn(idx)
    local object = data.object[idx]
end

local function COCOLoadImgKeypointsFn(idx)
  
end



-------------------------------------------------------------------------------
-- Select data loading function
-------------------------------------------------------------------------------

-- load corresponding set
local loadImgKeypointsFn
if opt.dataset == 'mpii' then
    loadImgKeypointsFn = MPIILoadImgKeypointsFn
elseif opt.dataset == 'flic' then
    loadImgKeypointsFn = FLICLoadImgKeypointsFn
elseif opt.dataset == 'lsp' then
    loadImgKeypointsFn = LSPLoadImgKeypointsFn
elseif opt.dataset == 'mscoco' or opt.dataset == 'coco' then
    loadImgKeypointsFn = COCOLoadImgKeypointsFn
else
    error('Invalid dataset: ' .. opt.dataset..'. Please use one of the following datasets: mpii, flic, lsp, mscoco.')
end


-------------------------------------------------------------------------------
-- Transform image
-------------------------------------------------------------------------------

local function ImageTransform(img, opt, mode)
    local output = img:clone()
    local is_flipped = false
    
    -- apply image transformations
    
    output = t.Fix()(output) -- fix channels
    output = t.Scale(opt.imageSize)(output) -- scale
    
    
    if mode == 'train' then
        output = t.RandomCrop(opt.inputRes)(output) -- random crop
        --output = t.ColorNormalize({mean = opt.meanstd.mean, std = opt.meanstd.std})(output)
        if torch.uniform() > 0.5 then
            output =  t.HorizontalFlip(1)(output)
            is_flipped = true
        end
        
    else
        output = t.CenterCrop(opt.cropSize)(output)
        --output = t.ColorNormalize({mean = opt.meanstd.mean, std = opt.meanstd.std})(output)
        
    end
    
    return output, is_flipped
end


-------------------------------------------------------------------------------
-- Load data for train/test
-------------------------------------------------------------------------------

function loadData(data, idx, mode)
  
    -- inits
    local r = 0 -- set rotation to 0
    
    -- Load image + keypoints + other data
    local index = idx
    if mode == 'train' then
        index = torch.random(1, data.object:size(1))
    end
    local img, keypoints, c, s, nJoints = loadImgKeypointsFn(data, index)
    
    -- Do rotation + scaling
    if mode == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
    end

    -- Crop image + craft heatmap
    local img_transf = crop2(img, c, s, r, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1,nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(heatmap[i], transform(torch.add(keypoints[i],1), c, s, r, opt.outputRes), 1)
        end
    end

    -- Do image augmentation/normalization
    -- subtract mean/std
    if mode == 'train' then
        -- Flipping
        if torch.uniform() < .5 then
            img_transf = flip(img_transf)
            heatmap = shuffleLR(flip(heatmap))
        end
        -- color augmentation
        --img_transf = ColorJitter(opt)(img_transf)
        img_transf[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        img_transf[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        img_transf[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end
    
    -- output
    return img_transf, heatmap
end

