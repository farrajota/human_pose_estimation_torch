--[[
    Train human pose benchmark script for FLIC/MPII/Leeds/MSCOCO datasets.
--]]

require 'torch'
require 'paths'
require 'string'


--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

paths.dofile('configs_benchmark.lua')

-- set local vars
local lopt = opt
local dataset = g_dataset
local nSamples = g_dataset[opt.setname].object:size(1)

-- load torchnet package
local tnt = require 'torchnet'


--paths.dofile('data.lua')
--local data = loadDataBenchmark(1, dataset.val, 'val')
--aqui=1

--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        init    = function(threadid) 
                    require 'torch'
                    require 'torchnet'
                    opt = lopt
                    paths.dofile('data.lua')
                    torch.manualSeed(threadid+opt.manualSeed)
                  end,
        closure = function()
          
            -- setup data
            local data = dataset[mode]
          
            -- setup dataset iterator
            return tnt.ListDataset{
                list = torch.range(1, nSamples):long(),
                load = function(idx)
                    local input, parts, center, scale, normalize = loadDataBenchmark(idx, data, mode)
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
    if opt.predictions == 0 then
        print('\n*********************************************************')
        print(('Start testing/benchmarking the network on the %s dataset: '):format(opt.dataset))
        print('*********************************************************')
    else
        print('\n***************************************************')
        print(('Start processing predictions for the %s dataset: '):format(opt.dataset))
        print('***************************************************')
    end
end


-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets, center, scale, normalize

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
    
    if opt.predictions == 0 then
        -- compute the distance accuracy
        
        -- Get predictions (hm and img refer to the coordinate space)
        local preds_hm, preds_img = getPredsBenchmark(output, center, scale)
        
        -- compute distances
        table.insert(distances, calcDistsOne(preds_img:squeeze(), parts, normalize):totable())
    end
    
    collectgarbage()
end


engine.hooks.onEnd= function(state)
  
    if opt.predictions == 0 then
        local labels = {'Validation'}
        local res = {}
        local dists = {torch.FloatTensor(distances):transpose(1,2)}
        
        -- plot benchmark results
        require 'gnuplot'
        gnuplot.raw('set bmargin 1')
        gnuplot.raw('set lmargin 3.2')
        gnuplot.raw('set rmargin 2')
        if opt.dataset=='mpii' then
            gnuplot.raw(('set multiplot layout 2,3 title "%s Validation Set Performance (PCKh@%.1f)"'):format(string.upper(opt.dataset), opt.threshold))
        else
            gnuplot.raw(('set multiplot layout 2,3 title "%s Validation Set Performance (PCK@%.1f)"'):format(string.upper(opt.dataset), opt.threshold))
        end
          
        gnuplot.raw('set xtics font ",6"')
        gnuplot.raw('set ytics font ",6"')
        
        if opt.dataset == 'mpii' then
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {9,10}, labels, 'Head', opt.threshold), 'Head'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {2,5}, labels, 'Knee', opt.threshold), 'Knee'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {1,6}, labels, 'Ankle', opt.threshold), 'Ankle'})
            print('-----------------------------------')
            gnuplot.raw('set tmargin 2.5')
            gnuplot.raw('set bmargin 1.5')
            table.insert(res, {displayPCK(dists, {13,14}, labels, 'Shoulder', opt.threshold), 'Shoulder'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {12,15}, labels, 'Elbow', opt.threshold), 'Elbow'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {11,16}, labels, 'Wrist', opt.threshold, true), 'Wrist'})
            print('-----------------------------------')
            
        elseif opt.dataset == 'flic' then
            table.insert(res, {displayPCK(dists, {1,4}, labels, 'Shoulder', opt.threshold), 'Shoulder'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {2,5}, labels, 'Elbow', opt.threshold), 'Elbow'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {3,6}, labels, 'Wrist', opt.threshold, true), 'Wrist'})
            print('-----------------------------------')
            
        elseif opt.dataset == 'lsp' then
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {13,14}, labels, 'Head', opt.threshold), 'Head'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {3,4}, labels, 'Hip', opt.threshold), 'Hip'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {2,5}, labels, 'Knee', opt.threshold), 'Knee'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {1,6}, labels, 'Ankle', opt.threshold), 'Ankle'})
            print('-----------------------------------')
            gnuplot.raw('set tmargin 2.5')
            gnuplot.raw('set bmargin 1.5')
            table.insert(res, {displayPCK(dists, {9,10}, labels, 'Shoulder', opt.threshold), 'Shoulder'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {8,11}, labels, 'Elbow', opt.threshold), 'Elbow'})
            print('-----------------------------------')
            table.insert(res, {displayPCK(dists, {7,12}, labels, 'Wrist', opt.threshold, true), 'Wrist'})
            print('-----------------------------------')
        elseif opt.dataset == 'mscoco' or opt.dataset == 'coco' then
        else
      end
      
        gnuplot.raw('unset multiplot')
        gnuplot.pngfigure(paths.concat(opt.save, 'Validation_Set_Performance_PCKh.png')) 
        gnuplot.plotflush()
        
        json.save(paths.concat(opt.save,'Validation_Set_Performance_results.json'), json.encode(res))
        
    else
        -- store predictions into disk
        torch.save(paths.concat(opt.save,'Predictions.t7'), predictions)
    end

end


--------------------------------------------------------------------------------
-- Benchmark/process predictions
--------------------------------------------------------------------------------

engine:test{
    network  = model,
    iterator = (opt.predictions==0 and getIterator('val')) or getIterator('test')
}

if opt.predictions == 0 then
    print('\nBenchmark script complete.')
else
    print('\nPredictions script complete.')
end