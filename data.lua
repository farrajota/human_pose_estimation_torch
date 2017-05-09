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
    center[2] = center[2] --+ 15 * scale
    scale = scale --* 1.25
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize 
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
    local kps_coords_x = keypoints:select(2,1)
    local kps_coords_y = keypoints:select(2,2)
    local kps_x=kps_coords_x[kps_coords_x:gt(0)]
    local kps_y=kps_coords_y[kps_coords_y:gt(0)]
    --local xmin = kps_x:min()
    --local xmax = kps_x:max()
    --local ymin = kps_y:min() 
    --local ymax = kps_y:max()
    --local keypoints_remove_zeros = keypoints[keypoints:gt(0):byte()]:view(-1,3) -- remove coords equal to 0
    --local xmin = keypoints_remove_zeros[{{},{1}}]:min()
    --local xmax = keypoints_remove_zeros[{{},{1}}]:max()
    --local ymin = keypoints_remove_zeros[{{},{2}}]:min()
    --local ymax = keypoints_remove_zeros[{{},{2}}]:max()
    local center = torch.FloatTensor({(kps_x:min()+kps_x:max())/2, (kps_y:min()+kps_y:max())/2})
    --local center = ((keypoints[10]+keypoints[3])/2):sub(1,2):squeeze()
    --local center = ((keypoints[4]+keypoints[3])/2):sub(1,2):squeeze()
    local scale = 1-- 1.25
    local normalize = (keypoints[10]-keypoints[3]):norm()
    
    -- Load image
    local img = image.load(filename,3,'float')
    
    --local center = torch.FloatTensor({img:size(3)/2, img:size(2)/2})
    
    -- image, keypoints, center coords, scale, number of joints
    return img, keypoints, center, scale, nJoints, normalize
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

-- MPII+LSP+LSPe
local function MPIILSPLoadImgKeypointsFn(data, idx)
    if data.isTrain then
        if idx <= data.data[1].object:size(1) then
            -- mpii train
            local img, keypoints, center, scale, nJoints, normalize = MPIILoadImgKeypointsFn(data.data[1], idx)
            return img, keypoints:index(1,torch.LongTensor({1,2,3,4,5,6,11,12,13,14,15,16,9,10})), center, scale, 14, normalize
        elseif idx <= data.data[1].object:size(1)+data.data[2].object:size(1) then
            -- lsp train
            return LSPLoadImgKeypointsFn(data.data[2], idx-data.data[1].object:size(1))
        else
            -- lspe train
            return LSPLoadImgKeypointsFn(data.data[3], idx-(data.data[1].object:size(1)+data.data[2].object:size(1)))
        end
    else
        return LSPLoadImgKeypointsFn(data, idx)
    end
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
elseif opt.dataset == 'mpii+lsp' then
    loadImgKeypointsFn = MPIILSPLoadImgKeypointsFn
else
    error('Invalid dataset: ' .. opt.dataset..'. Please use one of the following datasets: mpii, flic, lsp, mscoco.')
end
loadImgKeypointsFn_ = loadImgKeypointsFn


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
            --drawGaussian(heatmap[i], mytransform(torch.add(keypoints[i],1), c, s, r, opt.outputRes), 1)
            drawGaussian(heatmap[i], mytransform(keypoints[i], c, s, r, opt.outputRes), opt.hmGauss or 1)
        end
    end
    
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
    local s_ = s
    
    -- Do rotation + scaling
    if mode == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        --s = s * torch.uniform(1-opt.scale, 1+opt.scale)
        r = rnd(opt.rotate)
        if torch.uniform() <= opt.rotRate then r = 0 end
    end

    -- Crop image + craft heatmap
    local img_transf = crop2(img, c, s, r, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1,nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            local new_kp = transform(keypoints[i], c, s, r, opt.outputRes)
            drawGaussian(heatmap[i], new_kp, opt.hmGauss)
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
    
    -- output
    return img_transf, heatmap
end


function loadData_new(idx, data, mode)
  
    -- inits
    local r = 0 -- set rotation to 0
    
    -- Load image + keypoints + other data
    local img, keypoints, c, s, nJoints = loadImgKeypointsFn(data, idx)  
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

    -- Crop image + craft heatmap
    local total1, total2 = 0,0
    local img_transf = mycrop(img, c, s, r, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1,nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            local new_kp = mytransform(keypoints[i], c, s, r, opt.outputRes)
            drawGaussian(heatmap[i], new_kp:int(), opt.hmGauss)
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