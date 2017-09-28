--[[
    Script for testing a human pose predictor.

    Available/valid datasets: FLIC, MPII, Leeds Sports or COCO.
--]]


require 'paths'
require 'torch'
require 'string'

local tnt = require 'torchnet'


--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
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
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        ordered = true,
        init    = function(threadid)
                    require 'torch'
                    require 'torchnet'
                    opt = lopt
                    paths.dofile('data.lua')
                    paths.dofile('sample_batch.lua')
                    torch.manualSeed(threadid+opt.manualSeed)
                  end,
        closure = function()

            -- setup data loader
            local data_loader = select_dataset_loader(opt.dataset, mode)
            local loader = data_loader[mode]

            -- setup dataset iterator
            return tnt.ListDataset{
                list = torch.range(1, nSamples):long(),
                load = function(idx)
                    local input, parts, center, scale, normalize = getSampleTest(loader, idx)
                    return {input, parts, center, scale, normalize}
                end
            }:batch(1, 'include-last')
        end,
    }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
    print('\n*********************************************************')
    print(('Start testing the network on the %s dataset: '):format(opt.dataset))
    print('*********************************************************')
end


-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets, center, scale, normalize, t_matrix
engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    inputs:resize(state.sample[1]:size() ):copy(state.sample[1])
    parts   = state.sample[2][1]
    center  = state.sample[3][1]
    scale   = state.sample[4][1]
    normalize = state.sample[5][1]

    state.sample.input  = inputs
end


local predictions, distances = {}, {}
local coords = torch.FloatTensor(2, num_keypoints, nSamples):fill(0)
engine.hooks.onForward= function(state)
    xlua.progress(state.t, nSamples)

    -- fetch predictions of body joints from the last output
    local output
    if type(state.network.output) == 'table' then
        local num_outputs = #state.network.output
        output = state.network.output[num_outputs][1]:float()
    else
        output = state.network.output[1]:float()
    end

    -- clamp negatives values to 0
    output[output:lt(0)] = 0


    -- store predictions into a table
    table.insert(predictions, {output, center, scale})

    -- compute the distance accuracy
    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img = getPredsBenchmark(output, center, scale)

    -- compute distances
    local accuracy = calcDistsOne(preds_img:squeeze(), parts, normalize)
    table.insert(distances, accuracy:totable())

    -- add coordinates to the coords tensor
    coords[{{},{},{state.t}}] = preds_img:transpose(2,3):squeeze()

    collectgarbage()
end


engine.hooks.onEnd= function(state)
    local labels = {'Validation'}
    local res = {}
    local dists = {torch.FloatTensor(distances):transpose(1,2)}

    -- plot benchmark results
    require 'gnuplot'
    gnuplot.raw('set bmargin 1')
    gnuplot.raw('set lmargin 3.2')
    gnuplot.raw('set rmargin 2')
    if opt.dataset=='mpii' then
        gnuplot.raw(('set multiplot layout 2,3 title "%s Validation Set Performance (PCKh@%.1f)"'):format(string.upper(opt.dataset), opt.pck_threshold))
    else
        gnuplot.raw(('set multiplot layout 2,3 title "%s Validation Set Performance (PCK@%.1f)"'):format(string.upper(opt.dataset), opt.pck_threshold))
    end

    gnuplot.raw('set xtics font ",6"')
    gnuplot.raw('set ytics font ",6"')

    print('')
    print('-----------------------------------')
    print('PCK@' .. opt.pck_threshold)
    if opt.dataset == 'mpii' then
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {9,10}, labels, 'Head', opt.pck_threshold), 'Head'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {2,5}, labels, 'Knee', opt.pck_threshold), 'Knee'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {1,6}, labels, 'Ankle', opt.pck_threshold), 'Ankle'})
        print('-----------------------------------')
        gnuplot.raw('set tmargin 2.5')
        gnuplot.raw('set bmargin 1.5')
        table.insert(res, {displayPCK(dists, {13,14}, labels, 'Shoulder', opt.pck_threshold), 'Shoulder'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {12,15}, labels, 'Elbow', opt.pck_threshold), 'Elbow'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {11,16}, labels, 'Wrist', opt.pck_threshold, true), 'Wrist'})
        print('-----------------------------------')

    elseif opt.dataset == 'flic' then
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {1,4}, labels, 'Shoulder', opt.pck_threshold), 'Shoulder'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {2,5}, labels, 'Elbow', opt.pck_threshold), 'Elbow'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {3,6}, labels, 'Wrist', opt.pck_threshold, true), 'Wrist'})
        print('-----------------------------------')

    elseif opt.dataset == 'lsp' then
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {13,14}, labels, 'Head', opt.pck_threshold), 'Head'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {3,4}, labels, 'Hip', opt.pck_threshold), 'Hip'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {2,5}, labels, 'Knee', opt.pck_threshold), 'Knee'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {1,6}, labels, 'Ankle', opt.pck_threshold), 'Ankle'})
        print('-----------------------------------')
        gnuplot.raw('set tmargin 2.5')
        gnuplot.raw('set bmargin 1.5')
        table.insert(res, {displayPCK(dists, {9,10}, labels, 'Shoulder', opt.pck_threshold), 'Shoulder'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {8,11}, labels, 'Elbow', opt.pck_threshold), 'Elbow'})
        print('-----------------------------------')
        table.insert(res, {displayPCK(dists, {7,12}, labels, 'Wrist', opt.pck_threshold, true), 'Wrist'})
        print('-----------------------------------')
    elseif opt.dataset == 'coco' then
        -- TODO
    else
        error(('Invalid dataset name: %s. Available datasets: mpii | flic | lsp | coco'):format(name))
    end

    gnuplot.raw('unset multiplot')
    gnuplot.pngfigure(paths.concat(opt.save, 'Validation_Set_Performance_PCKh.png'))
    gnuplot.plotflush()

    json.save(paths.concat(opt.save,'Validation_Set_Performance_results.json'), json.encode(res))
end


--------------------------------------------------------------------------------
-- Test the model
--------------------------------------------------------------------------------

engine:test{
    network  = model,
    iterator = getIterator('test')
}

print('\nTest script complete.')