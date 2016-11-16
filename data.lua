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
    local head_coord = data.head_coord[object[6]]
    local normalize = torch.FloatTensor({head_coord[3]-head_coord[1], head_coord[4]-head_coord[2]}):norm() * 0.6 
    
    -- Small adjustment so cropping is less likely to take feet out
    center[2] = center[2] + 15 * scale
    scale = scale * 1.25
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints:narrow(2,1,2), center, scale, nJoints, normalize 
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
    local normalize = torch.FloatTensor({torso[3]-torso[1], torso[4]-torso[2]}):norm()
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
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
    local normalize = torch.FloatTensor({xmax-xmin, ymax-ymin}):norm()
    
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

function loadDataBenchmark(idx, data, mode)
  
    -- inits
    local r = 0 -- set rotation to 0
    
    -- Load image + keypoints + other data
    local img, keypoints, c, s, nJoints, normalize = loadImgKeypointsFn(data, idx)

    -- Crop image + craft heatmap
    local img_transf = crop2(img, c, s, r, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1,nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(heatmap[i], transform(torch.add(keypoints[i],1), c, s, r, opt.outputRes), 1)
        end
    end

    --local meanstd = {mean={0.25607767449153,0.23222393443673, 0.2148381369845}, std={0.2086786306043,0.19715121058158,0.19158156380364}}
    
    --for i=1, 3 do
    --    img_transf[i]:add(-meanstd.mean[i]):div(meanstd.std[i])
    --end
    

    -- output
    -- input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), c, s, normalize
end


-------------------------------------------------------------------------------
-- Load data for one object in the train/test data
-------------------------------------------------------------------------------

function loadData(idx, data, mode)
  
    -- inits
    local r = 0 -- set rotation to 0
    
    -- Load image + keypoints + other data
    local img, keypoints, c, s, nJoints = loadImgKeypointsFn(data, idx)
    
    if opt.centerjit > 0 then
        local offset = c:clone():random(-opt.centerjit, opt.centerjit)
        c = c:add(offset)
    end
    
    
    -- Do rotation + scaling
    if mode == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        --if torch.uniform() <= .6 then r = 0 end
        if torch.uniform() <= opt.rotRate then r = 0 end
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
        if opt.colourjit then
            local opts_jit = {brightness = 0.4,contrast = 0.4,saturation = 0.4}
            img_transf = t.ColorJitter(opts_jit)(img_transf)
        else
            img_transf[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
            img_transf[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
            img_transf[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        end
    end
    
    if opt.pca then
        local pca = {
           eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
           eigvec = torch.Tensor{
              { -0.5675,  0.7192,  0.4009 },
              { -0.5808, -0.0045, -0.8140 },
              { -0.5836, -0.6948,  0.4203 },
           },
        }
        img_transf = t.Lighting(0.1, pca.eigval, pca.eigvec)(img_transf)
    end
    
    
    -- normalize mean/std
    if opt.colourNorm then
        img_transf = t.ColorNormalize(opt.meanstd)(img_transf)
    end
    
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
        local idx = torch.random(1, data.object:size(1))
        table.insert(sample, {loadData(idx, data, mode)})
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