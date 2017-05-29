--[[
    Train + benchmark the 'best' network on the flic dataset.
]]


require 'paths'
require 'torch'


--------------------------------------------------------------------------------
-- Train + benchmark network
--------------------------------------------------------------------------------

do
    local info = {
        -- experiment id
        expID = 'final-best',
        netType = 'hg-generic-best',
        dataset = 'flic',

        -- data augment
        colourNorm = 'false',
        colourjit = 'true',
        centerjit = 0,
        pca = 'false',
        dropout = 0,
        spatialdropout = 0.2,
        critweights = 'none',
        scale = 0.30,
        rotate = 40,
        rotRate = 0.2,

        -- train options
        optMethod = 'adam',
        nThreads = 2,
        trainIters = 1000,
        nStack = 8,
        nFeats = 256,
        schedule = "{{50,2.5e-4,0},{15,1e-4,0},{10,5e-5,0}}",
        batchSize = 4,
        snapshot = 10,
        nGPU = 2,
        saveBest = 'true',
        continue = 'false',
    }

    -- concatenate options fields to a string
    local str_args = ''
    for k, v in pairs(info) do
        str_args = str_args .. ('-%s %s '):format(k, v)
    end

    local str_cuda
    if info.nGPU <= 1 then
        str_cuda = 'CUDA_VISIBLE_DEVICES=1'
    else
        str_cuda = 'CUDA_VISIBLE_DEVICES=1,0'
    end

    -- display options
    print('Input options: ' .. str_args)

    -- train network
    os.execute(('%s th train.lua %s'):format(str_cuda, str_args))

    -- benchmark network
    os.execute('CUDA_VISIBLE_DEVICES=1 th benchmark.lua ' .. str_args)
end


--------------------------------------------------------------------------------
-- Train + benchmark ensemble network
--------------------------------------------------------------------------------

do
    local info = {
        -- experiment id
        expID = 'final-ensemble-best',
        netType = 'hg-generic-ensemble',
        ensembleID = 'final-best',

        -- data augment
        colourNorm = 'false',
        colourjit = 'true',
        centerjit = 0,
        pca = 'false',
        dropout = 0,
        spatialdropout = 0.2,
        critweights = 'none',
        scale = 0.3,
        rotate = 40,
        rotRate = 0.2,

        -- train options
        optMethod = 'adam',
        nThreads = 2,
        trainIters = 1000,
        nStack = 8,
        nFeats = 256,
        schedule = "\'{{15,1e-3,0},{5,1e-4,0},{5,5e-5,0}}\'",
        batchSize = 8,
        snapshot = 0,
        nGPU = 2,
        saveBest = 'true',
        continue = 'false',
    }

    -- concatenate options fields to a string
    local str_args = ''
    for k, v in pairs(info) do
        str_args = str_args .. ('-%s %s '):format(k, v)
    end

    local str_cuda
    if info.nGPU <= 1 then
        str_cuda = 'CUDA_VISIBLE_DEVICES=1'
    else
        str_cuda = 'CUDA_VISIBLE_DEVICES=1,0'
    end

    -- display options
    print('Input options: ' .. str_args)

    -- train network
    os.execute(('%s th train.lua %s'):format(str_cuda, str_args))

    -- benchmark network
    os.execute('CUDA_VISIBLE_DEVICES=1 th benchmark.lua ' .. str_args)
end