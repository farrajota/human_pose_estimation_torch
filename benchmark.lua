--[[
    Produce predictions for a dataset.

    The benchmark algorithm only works for the FLIC and LSP datasets.
    The MPII and COCO datasets have a dedicated online server for evaluation.
--]]


require 'paths'
require 'torch'
require 'string'

local tnt = require 'torchnet'


--------------------------------------------------------------------------------
-- Load configs
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
                    local input, center, scale, normalize = getSampleBenchmark(loader, idx)
                    return {input, center, scale, normalize}
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
    print('\n***************************************************')
    print(('Start processing predictions for the %s dataset: '):format(opt.dataset))
    print('***************************************************')
end


-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets, center, scale, normalize, t_matrix
engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    inputs:resize(state.sample[1]:size() ):copy(state.sample[1])
    center  = state.sample[2][1]
    scale   = state.sample[3][1]
    normalize = state.sample[4][1]

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

    -- add coordinates to the coords tensor
    coords[{{},{},{state.t}}] = preds_img:transpose(2,3):squeeze()

    collectgarbage()
end


engine.hooks.onEnd= function(state)
    print('Storing predictions to disk.. ')
    local matio = require 'matio'
    torch.save(paths.concat(opt.save,'Predictions.t7'), {pred=coords})
    matio.save(paths.concat(opt.save,'Predictions.mat'),{pred=coords:double()})
end


--------------------------------------------------------------------------------
-- process predictions
--------------------------------------------------------------------------------

engine:test{
    network  = model,
    iterator = getIterator('test')
}

print('\nPredictions script complete.')


--------------------------------------------------------------------------------
-- Benchmark algorithm
--------------------------------------------------------------------------------

local str = string.lower(opt.dataset)
if str == 'flic' or str == 'lsp'  then
    local benchmark_folder = paths.concat(projectDir, 'human-pose-benchmark')

    -- setup algorithm folder
    local bench_alg_path = paths.concat(benchmark_folder, 'algorithms', opt.eval_plot_name)
    if not paths.dirp(bench_alg_path) then
        print('Saving everything to: ' .. bench_alg_path)
        os.execute('mkdir -p ' .. bench_alg_path)
        os.execute(('echo \'Ours\' > %s'):format(paths.concat(bench_alg_path, 'algorithm.txt')))
    end

    -- rename predictions file
    local filename_old = paths.concat(opt.save,'Predictions.mat')
    local filename_new
    if str == 'flic' then
        filename_new = paths.concat(bench_alg_path, 'pred_keypoints_flic_oc.mat')
    else
        filename_new = paths.concat(bench_alg_path, 'pred_keypoints_lsp_pc.mat')
    end
    os.execute(('cp %s %s'):format(filename_old, filename_new))

    -- process benchmark
    local command = ('cd %s && matlab -nodisplay -nodesktop -r "try, %s, catch, exit, end, exit"')
                    :format(benchmark_folder, 'benchmark_' .. str)
    os.execute(command)

    -- copy plots folder to the experiment dir
    os.execute(('cp -ar %s %s'):format(paths.concat(benchmark_folder, 'plots'), paths.concat(opt.save, 'plots')))
elseif str == 'mpii' or str == 'coco' then
    -- TODO
else
    error(('Invalid dataset name: %s. Available datasets: mpii | flic | lsp | coco'):format(name))
end