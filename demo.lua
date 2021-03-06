--[[
    Human pose estimation demo.

    This script randomly samples 'N' images from a dataset and
    displays the predictions of body joint of a network.

    Warning: requires either 'qlua' or 'display' module to work.
]]


require 'paths'
require 'torch'
require 'string'


if not pcall(require, 'qt') then
    if pcall(require, 'display') then
        display = require 'display'
    else
        error('Run this script using \'qlua\' or install the \'display\' package with luarocks.')
    end
end


--------------------------------------------------------------------------------
-- Load configs (data generator, model)
--------------------------------------------------------------------------------

paths.dofile('configs.lua')

-- load model from disk
load_model('test')

-- set local vars
local lopt = opt
local nSamples, num_keypoints
do
    local mode = 'test'
    local data_loader = select_dataset_loader(opt.dataset, mode)
    local loader = data_loader[mode]
    nSamples = loader.size
    num_keypoints = loader.num_keypoints
end

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end


--------------------------------------------------------------------------------
-- Process examples
--------------------------------------------------------------------------------

print('\n==============================================')
print(('Selected dataset: %s'):format(opt.dataset))
print('==============================================\n')

print('Processing image predictions...')

local mode = 'test'

-- setup data loader
local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]


-- alloc memory in the gpu for faster data transfer
local input = torch.Tensor(1, 3, opt.inputRes, opt.inputRes); input=cast(input)

local randperm = torch.randperm(loader.size)
for i = 1,opt.demo_nsamples do
    -- get random image
    local idx = randperm[i] --torch.random(1, loader.size) -- get random idx
    local im, center, scale, _ = getSampleBenchmark(loader, idx)

    -- process body joint predictions
    input[1]:copy(im) -- copy data from CPU to GPU
    local out = model:forward(input)
    local hm = out[#out][1]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img = getPredsBenchmark(hm, center, scale)

    -- Display the result
    --preds_hm:mul(4) -- Change to input scale

    -- local dispImg = drawOutput(im, hm, preds_hm[1])
    local heatmaps = drawImgHeatmapParts(im, hm)

    -- get crop window
    local crop_coords = im[1]:gt(0):nonzero()
    local center_x_slice = im[{{1}, {opt.inputRes/2}, {}}]:squeeze():gt(0):nonzero()
    local x1, x2 = center_x_slice:min(), center_x_slice:max()
    local center_y_slice = im[{{1}, {}, {opt.inputRes/2}}]:squeeze():gt(0):nonzero()
    local y1, y2 = center_y_slice:min(), center_y_slice:max()
    local skeliImg = drawSkeleton(im, hm, torch.mul(preds_hm[1], opt.inputRes/opt.outputRes))
    local heatmaps_disp = {image.crop(skeliImg, x1, y1, x2, y2)}  --{image.crop(im, x1, y1, x2, y2)}
    for i=1, #heatmaps do
        table.insert(heatmaps_disp, image.crop(image.scale(heatmaps[i], opt.inputRes),x1,y1,x2,y2))
    end

    if opt.demo_plot_screen then
        if pcall(require, 'qt') then
            image.display({image = heatmaps_disp, title='heatmaps_image_'..i})
        else
            display.image(heatmaps_disp, {title='heatmaps_image_'..i})
        end
    end


    if opt.demo_plot_networks_predictions then
        ae_outputs = {image.crop(im, x1, y1, x2, y2)}
        for k=1, #out do
            local ae_predictions = out[k][1]:float()
            ae_predictions[ae_predictions:lt(0)] = 0
            local ae_heatmap_concat = ae_predictions:sum(1):squeeze()
            local ae_heatmap_color = drawImgHeatmapSingle(im, ae_heatmap_concat)
            table.insert(ae_outputs, image.crop(ae_heatmap_color,x1,y1,x2,y2))
        end

        if opt.demo_plot_screen then
            if pcall(require, 'qt') then
                image.display({image = ae_outputs, title='auto-encoders outputs heatmaps_image_'..i})
            else
                display.image(ae_outputs, {title='auto-encoders outputs heatmaps_image_'..i})
            end
        end
    end


    if opt.demo_plot_save then
        local plot_dir = 'image_plots'
        if not paths.dirp(paths.concat(opt.save, plot_dir)) then
            print('Saving plots to: ' .. paths.concat(opt.save, plot_dir))
            os.execute('mkdir -p ' .. paths.concat(opt.save, plot_dir))
        end
        if not paths.dirp(paths.concat(opt.save, plot_dir, 'skeletons_all')) then
            print('Saving plots to: ' .. paths.concat(opt.save, plot_dir, 'skeletons_all'))
            os.execute('mkdir -p ' .. paths.concat(opt.save, plot_dir, 'skeletons_all'))
        end
        if not paths.dirp(paths.concat(opt.save, plot_dir, idx)) then
            print('Saving plots to: ' .. paths.concat(opt.save, plot_dir, idx))
            os.execute('mkdir -p ' .. paths.concat(opt.save, plot_dir, idx))
        end
        image.save(paths.concat(opt.save, plot_dir, idx, 'image.png'), image.crop(im, x1, y1, x2, y2))
        image.save(paths.concat(opt.save, plot_dir,idx, 'skeleton.png'), image.crop(skeliImg, x1, y1, x2, y2))
        image.save(paths.concat(opt.save, plot_dir, 'skeletons_all', 'skeleton_'..idx..'.png'), image.crop(skeliImg, x1, y1, x2, y2))
        for j=2, #heatmaps_disp do
            image.save(paths.concat(opt.save, plot_dir, idx, 'heatmap_'..(j-1)..'.png'), heatmaps_disp[j])
        end

        if opt.demo_plot_networks_predictions then
            for j=2, #ae_outputs do
                image.save(paths.concat(opt.save, plot_dir, idx, 'AE_'..(j-1)..'.png'), ae_outputs[j])
            end
        end
    end

    xlua.progress(i, opt.demo_nsamples)
    collectgarbage()
end

print('Demo script complete.')