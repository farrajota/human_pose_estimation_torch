--[[
    Train the 'hg-stacked' network arch on the FLIC dataset.

    Info:

        Network name/id/type: stacked (hourglass).

        Train Ensemble: no.

        Dataset: FLIC.
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
        expID = 'hg-stacked',
        netType = 'hg-stacked',
        dataset = 'flic',

        -- data augment
        colourNorm = 'false',
        colourjit = 'true',
        centerjit = 0,
        pca = 'false',
        dropout = 0,
        spatialdropout = 0,
        critweights = 'none',
        scale = 0.30,
        rotate = 40,
        rotRate = 0.2,

        -- train options
        optMethod = 'adam',
        nThreads = 2,
        trainIters = 1000,
        testIters = 300,
        nStack = 2,
        nFeats = 256,
        schedule = "{{50,2.5e-4,0},{15,1e-4,0},{10,5e-5,0}}",
        batchSize = 4,
        snapshot = 10,
        nGPU = 1,
        saveBest = 'true',
        continue = 'false',
        clear_buffers = 'true',
    }

    -- concatenate options fields to a string
    local str_args = ''
    for k, v in pairs(info) do
        str_args = str_args .. ('-%s %s '):format(k, v)
    end

    local str_cuda
    if info.nGPU <= 1 then
        str_cuda = 'CUDA_VISIBLE_DEVICES=0'
    else
        str_cuda = 'CUDA_VISIBLE_DEVICES=0,1'
    end

    -- train network
    exec_command(('%s th train.lua %s'):format(str_cuda, str_args))
end