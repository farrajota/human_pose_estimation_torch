--[[
    Data sampling functions.
]]


require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
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

-- mpii
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

-- flic
local function FLICLoadImgKeypointsFn(data, idx)
    local object = data.object[idx]
    local filename = ffi.string(data.filename[object[1]]:data())
    local keypoints = data.keypoint[object[3]]
    local torso = data.torso[object[4]]
    local center = torch.FloatTensor({(torso[1]+torso[3])/2, (torso[2]+torso[4])/2})
    local scale = 2.2
    local nJoints = keypoints:size(1)/3
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints:view(nJoints, 3), center, scale, nJoints
end

-- leeds sports
local function LSPLoadImgKeypointsFn(data, idx)
    local object = data.object[idx]
    local filename = ffi.string(data.filename[object[1]]:data())
    local keypoints = data.keypoint[object[3]]
    local nJoints = keypoints:size(1)/3
    keypoints = keypoints:view(nJoints, 3)
    
    -- calc center coordinates
    local xmin = keypoints[{{},{1}}]:min()
    local xmax = keypoints[{{},{1}}]:max()
    local ymin = keypoints[{{},{2}}]:min()
    local ymax = keypoints[{{},{2}}]:max()
    local center = torch.FloatTensor({(xmin+xmax)/2, (ymin+ymax)/2})
    local scale = 1.0
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints, center, scale, nJoints
end

-- ms coco keypoints
local function COCOLoadImgKeypointsFn(data, idx)
    local object = data.object[idx]
    local filename = ffi.string(data.filename[object[1]]:data())
    local keypoints = data.keypoint[object[4]]
    local nJoints = keypoints:size(1)/3
    keypoints = keypoints:view(nJoints, 3)
    
    -- calc center coordinates
    local bbox = data.bbox[object[3]]
    local center = torch.FloatTensor({(bbox[1]+bbox[3])/2, (bbox[2]+bbox[4])/2})
    local bbox_width = bbox[3]-bbox[1]
    local bbox_height = bbox[4]-bbox[2]
    local scale = math.max(bbox_height, bbox_width)/200 * 1.25
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints, center, scale, nJoints
  
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
-- Load data for one object in the train/test data
-------------------------------------------------------------------------------

function loadData(data, mode)
  
    -- inits
    local r = 0 -- set rotation to 0
    
    -- Load image + keypoints + other data
    local idx = torch.random(1, data.object:size(1))
    local img, keypoints, c, s, nJoints = loadImgKeypointsFn(data, idx)
    
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
    
    -- normalize mean/std
    --img_transf = t.ColorNormalize(opt.meanstd)(img_transf)
    
    -- output
    return img_transf, heatmap
end


-------------------------------------------------------------------------------
-- Get a batch of data samples
-------------------------------------------------------------------------------

function getSampleBatch(data, mode, batchSize)
    local sample = {}
    local batchSize = batchSize or opt.batchSize
    
    -- get batch data
    for i=1, batchSize do
        table.insert(sample, {loadData(data, mode)})
    end
    
    -- concatenate data
    local imgs_tensor = torch.FloatTensor(batchSize, sample[1][1]:size(1), sample[1][1]:size(2), sample[1][1]:size(3)):fill(0)
    local heatmaps_tensor = torch.FloatTensor(batchSize, sample[1][2]:size(1), sample[1][2]:size(2), sample[1][2]:size(3)):fill(0)
    
    for i=1, batchSize do
        imgs_tensor[i]:copy(sample[i][1])
        heatmaps_tensor[i]:copy(sample[i][2])
    end
    
    collectgarbage()
    
    return imgs_tensor, heatmaps_tensor
end


-------------------------------------------------------------------------------
-- Compute mean/std for the dataset
-------------------------------------------------------------------------------

function ComputeMeanStd(data)
    print('Preparing train val and meanstd cache.')
    
    local tnt = require 'torchnet'
    
    local nSamples = 1000
    print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   
    -- setup dataset iterator
    local iter = tnt.ListDataset{  -- replace this by your own dataset
      list = torch.range(1, nSamples):long(),
      load = function(idx)
          local input, _ = getSampleBatch(data, 'train', 1)
          return input[1]
      end
    }:iterator()

    local tm = torch.Timer()
    local meanEstimate = {0,0,0}
    local idx = 1
    xlua.progress(0, nSamples)
    for img in iter() do
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
      idx = idx + 1
      if idx%50==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
    end
    local mean = meanEstimate
    
    
    print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
    local stdEstimate = {0,0,0}
    idx = 1
    xlua.progress(0, nSamples)
    for img in iter() do
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
      idx = idx + 1
      if idx%50==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
    end
    local std = stdEstimate

    local cache = {}
    cache.mean = mean
    cache.std = std
    
    print('Time to estimate:', tm:time().real)
    return cache
end