--[[
    Train the 'best' network arch on the COCO dataset (task: keypoints).
]]


local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end


--------------------------------------------------------------------------------
-- Train network
--------------------------------------------------------------------------------

do
    local info = {
        -- experiment id
        expID = 'final-best',
        netType = 'hg-generic-best',
        dataset = 'coco',

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
        trainIters = 10000,
        testIters = 4000,
        nStack = 8,
        nFeats = 256,
        schedule = "{{75,2.5e-4,0},{15,1e-4,0},{10,5e-5,0}}",
        batchSize = 4,
        snapshot = 25,
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

    -- train network
    exec_command(('%s th train.lua %s'):format(str_cuda, str_args))
end


--------------------------------------------------------------------------------
-- Train ensemble
--------------------------------------------------------------------------------

do
    local info = {
        -- experiment id
        expID = 'final-ensemble-best',
        netType = 'hg-generic-ensemble',
        ensembleID = 'final-best',
        dataset = 'coco',

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
        trainIters = 10000,
        testIters = 4000,
        nStack = 8,
        nFeats = 256,
        schedule = "{{15,1e-3,0},{5,1e-4,0},{5,5e-5,0}}",
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

    -- train network
    exec_command(('%s th train.lua %s'):format(str_cuda, str_args))
end