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
paths.dofile('util/Logger.lua')
paths.dofile('util/store.lua')
paths.dofile('util/draw.lua')
paths.dofile('util/utils.lua')

--local ffi = require 'ffi'
--local t = paths.dofile('transforms.lua')


-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x)
    return math.max(-2*x, math.min(2*x, torch.randn(1)[1]*x))
end


-------------------------------------------------------------------------------
--Image + keypoint loading functions
-------------------------------------------------------------------------------

local function get_db_loader(name)
    local dbc = require 'dbcollection.manager'

    local dbloader
    local str = string.lower(name)
    if str == 'flic' then
        dbloader = dbc.load{name='flic', task='keypoints_d', data_dir=opt.data_dir}
    elseif str == 'lsp' then
        dbloader = dbc.load{name='leeds_sports_pose_extended', task='keypoints_d', data_dir=opt.data_dir}
    elseif str == 'mpii' then
        dbloader = dbc.load{name='mpii_pose', task='keypoints_d', data_dir=opt.data_dir}
    elseif str == 'mscoco' then
        dbloader = dbc.load{name='mscoco', task='keypoint_2016_d', data_dir=opt.data_dir}
    else
        error(('Invalid dataset name: %s. Available datasets: mpii | flic | lsp | mscoco'):format(name))
    end
    return dbloader
end

------------------------------------------------------------------------------------------------------------

local function loader_flic(set_name)
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str

    local dbloader = get_db_loader('flic')

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx)
        local data = dbloader:object(set_name, idx, true)[1]

        local filename = paths.concat(dbloader.data_dir, ascii2str(data[1])[1])
        local keypoints = data[4]:float()
        local torso_bbox = data[3]:float():squeeze()
        local center = torch.FloatTensor({(torso_bbox[1]+torso_bbox[3])/2,
                                          (torso_bbox[2]+torso_bbox[4])/2})
        local scale = 2.2
        local normalize = torch.FloatTensor({torso_bbox[3]-torso_bbox[1],
                                             torso_bbox[4]-torso_bbox[2]}):norm()

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_lsp(set_name)
    local dbloader = get_db_loader('lsp')

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx)
        local data = dbloader:object(set_name, idx, true)

        local filename = paths.concat(dbloader.data_dir, data[1])
        local keypoints = data[2]
        local kps_coords_x = keypoints:select(2,1)
        local kps_coords_y = keypoints:select(2,2)
        local kps_x=kps_coords_x[kps_coords_x:gt(0)]
        local kps_y=kps_coords_y[kps_coords_y:gt(0)]
        local center = torch.FloatTensor({(kps_x:min() + kps_x:max()) / 2,
                                          (kps_y:min() + kps_y:max()) / 2})
        local scale = 1-- 1.25
        local normalize = (keypoints[10]-keypoints[3]):norm()

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_mpii(set_name)
    local dbloader = get_db_loader('mpii')

    -- split train set annotations into two. (train + val)
    -- TODO

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx, num_keypoints)
        local data = dbloader:object(set_name, idx, true)

        local filename = paths.concat(dbloader.data_dir, data[1])
        local keypoints = data[5]
        local head_coord = data[4]
        local center = data[3]
        local scale = data[2]
        local normalize = torch.FloatTensor({head_coord[3]-head_coord[1],
                                             head_coord[4]-head_coord[2]}):norm() * 0.6

        -- Small adjustment so cropping is less likely to take feet out
        center[2] = center[2] --+ 15 * scale
        scale = scale --* 1.25

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        if num_keypoints then
            -- This section is only used in conjunction with the lsp dataset
            local kps = keypoints:index(1,torch.LongTensor({1,2,3,4,5,6,11,12,13,14,15,16,9,10}))
            return img, kps, center, scale, num_keypoints, normalize
        else
            return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
        end
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_coco(set_name)
    local dbloader = get_db_loader('mscoco')

    if set_name == 'test' then
        set_name = 'val' -- use mscoco val set for testing
    end

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx)
        local data = dbloader:object(set_name, idx, true)

        local filename = paths.concat(dbloader.data_dir, data[1])
        local keypoints = data[10]

        -- calc center coordinates
        local bbox = data[4]
        local center = torch.FloatTensor({(bbox[1]+bbox[3])/2, (bbox[2]+bbox[4])/2})
        local bbox_width = bbox[3]-bbox[1]
        local bbox_height = bbox[4]-bbox[2]
        local scale = math.max(bbox_height, bbox_width)/200 * 1.25
        local normalize = torch.FloatTensor({bbox_width, bbox_height}):norm() * 0.6

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_mpii_lsp(set_name)
    local loader_mpii = loader_mpii(set_name)
    local loader_lsp = loader_lsp(set_name)

    -- number of samples per train/test sets
    local size_lsp = loader_lsp.size
    local size_mpii = loader_mpii.size
    local set_size = size_lsp + size_mpii

    -- number of keypoints (body joints)
    local nJoints = loader_lsp.num_keypoints

    local data_loader = function(idx)
        if idx > size_lsp then
            return loader_mpii.loader(idx - size_lsp, nJoints)
        else
            return loader_lsp.loader(idx)
        end
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_dataset(name, mode)
    local str = string.lower(name)
    if str == 'flic' then
        return loader_flic(mode)
    elseif str == 'lsp' then
        return loader_lsp(mode)
    elseif str == 'mpii' then
        return loader_mpii(mode)
    elseif str == 'mscoco' then
        return loader_coco(mode)
    elseif str == 'mpii+lsp' then
        return loader_mpii_lsp(mode)
    else
        error(('Invalid dataset name: %s. Available datasets: mpii | flic | lsp | mscoco | mpii+lsp.'):format(name))
    end
end

------------------------------------------------------------------------------------------------------------

function select_dataset_loader(name, mode)
    assert(name)
    assert(mode)

    local str = string.lower(mode)
    if str == 'train' then
        return {
            train = fetch_loader_dataset(name, 'train'),
            test = fetch_loader_dataset(name, 'test')
        }
    elseif str == 'test' then
        return {
            test = fetch_loader_dataset(name, 'test')
        }
    else
        error(('Invalid mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end


-------------------------------------------------------------------------------
-- Test: transform image
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

    -- output: input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), c, s, normalize
end



-------------------------------------------------------------------------------
-- Benchmark: transform image
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

    -- output: input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), c, s, normalize
end


-------------------------------------------------------------------------------
-- Transform data for one object sample
-------------------------------------------------------------------------------

function transform_data(img, keypoints, center, scale, nJoints)

    -- inits
    local rot = 0 -- set rotation to 0
    local s_ = scale

    -- Do rotation + scaling
    if mode == 'train' then
        -- Scale and rotation augmentation
        scale = scale * (2 ^ rnd(opt.scale))
        rot = rnd(opt.rotate)
        if torch.uniform() <= opt.rotRate then
            rot = 0
        end
    end

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1, nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            local new_kp = transform(keypoints[i], center, scale, rot, opt.outputRes)
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
            local opts_jit = {brightness = 0.4,
                              contrast = 0.4,
                              saturation = 0.4}
            img_transf = t.ColorJitter(opts_jit)(img_transf)
        else
            img_transf[1]:mul(torch.uniform(0.6, 1.4)):clamp(0,1)
            img_transf[2]:mul(torch.uniform(0.6, 1.4)):clamp(0,1)
            img_transf[3]:mul(torch.uniform(0.6, 1.4)):clamp(0,1)
        end
    end

    -- output
    return img_transf, heatmap
end

------------------------------------------------------------------------------------------------------------

--[[ Transform the data for accuracy evaluation ]]
function transform_data_test(img, keypoints, center, scale, nJoints)
    -- inits
    local rot = 0 -- set rotation to 0

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1, nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            --drawGaussian(heatmap[i], mytransform(torch.add(keypoints[i],1), c, s, r, opt.outputRes), 1)
            drawGaussian(heatmap[i], mytransform(keypoints[i], center, scale, rot, opt.outputRes), opt.hmGauss or 1)
        end
    end

    -- output: input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), center, scale, normalize
end


-------------------------------------------------------------------------------
-- Get a batch of data samples
-------------------------------------------------------------------------------

function getSampleBatch(data_loader, batchSize)
    local sample = {}
    local batchSize = batchSize or opt.batchSize or 1

    local size = data_loader.size

    -- get batch data
    for i=1, batchSize do
        local idx = torch.random(1, size)
        local img, keypoints, center, scale, nJoints = data_loader.loader(idx)
        local imgs_t, heatmaps_t = transform_data(img, keypoints, center, scale, nJoints)
        table.insert(sample, {imgs_t, heatmaps_t})
    end

    -- concatenate data
    local imgs_tensor = torch.FloatTensor(batchSize,
                                          sample[1][1]:size(1),
                                          sample[1][1]:size(2),
                                          sample[1][1]:size(3)):fill(0)
    local heatmaps_tensor = torch.FloatTensor(batchSize,
                                              sample[1][2]:size(1),
                                              sample[1][2]:size(2),
                                              sample[1][2]:size(3)):fill(0)

    for i=1, batchSize do
        imgs_tensor[i]:copy(sample[i][1])
        heatmaps_tensor[i]:copy(sample[i][2])
    end

    collectgarbage()

    return imgs_tensor, heatmaps_tensor
end

------------------------------------------------------------------------------------------------------------

function getSampleTest(data_loader, idx)

    -- set rotation to 0
    local rot = 0

    -- Load image + keypoints + other data
    local img, keypoints, center, scale, nJoints, normalize = data_loader.loader(idx)

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1,nJoints do
        -- Checks that there is a ground truth annotation
        if keypoints[i][1] > 1 then
            drawGaussian(heatmap[i], mytransform(keypoints[i], center, scale, rot, opt.outputRes), opt.hmGauss or 1)
        end
    end

    -- output: input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), center, scale, normalize
end

------------------------------------------------------------------------------------------------------------

function getSampleBenchmark(data_loader, idx)

    -- set rotation to 0
    local rot = 0

    -- Load image + keypoints + other data
    local img, keypoints, center, scale, nJoints, normalize = data_loader.loader(idx)

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)

    -- output: input, label, center, scale, normalize
    return img_transf, center, scale, normalize
end

-------------------------------------------------------------------------------
-- Compute mean/std for the dataset
-------------------------------------------------------------------------------

function ComputeMeanStd(loader)
    assert(loader)

    print('Preparing meanstd cache...')

    local tnt = require 'torchnet'

    local nSamples = 1000
    local batchSize = 1

    print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')

    -- setup dataset iterator
    local iter = tnt.ListDataset{  -- replace this by your own dataset
        list = torch.range(1, nSamples):long(),
        load = function(idx)
            local input, _ = getSampleBatch(loader, batchSize)
            return input[1]
        end
    }:iterator()

    local tm = torch.Timer()
    local meanEstimate = {0,0,0}
    local idx = 1
    xlua.progress(0, nSamples)
    for img in iter() do
        for j=1, 3 do
            meanEstimate[j] = meanEstimate[j] + img[j]:mean()
        end
        idx = idx + 1
        if idx%50==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1, 3 do
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

    local cache = {
        mean = mean,
        str = str
    }

    print('Time to estimate:', tm:time().real)
    return cache
end