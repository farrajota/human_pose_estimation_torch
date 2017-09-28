--[[
    This script trains a network on the FLIC dataset under some configurations.

    Info:

        Network name/id/type: stacked-aes-ensemble.

        Train Ensemble: yes.

        Dataset: FLIC.
]]


local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end


--------------------------------------------------------------------------------
-- Train stacked network
--------------------------------------------------------------------------------

local info = {
    expID = 'stacked-aes-ensemble',
    netType = 'stacked-aes-ensemble',
    dataset = 'flic',

    optMethod = 'adam',
    colourNorm = 'false',
    colourjit = 'true',
    centerjit = 0,
    pca = 'false',
    dropout = 0,
    spatialdropout = 0.2,
    critweights = 'none',

    scale = 0.25,
    rotate = 30,
    rotRate = 0.333,
    nStack = 8,
    nFeats = 256,
    --schedule = '{{150,1e-4,0}}',
    --schedule = '{{50,2.5e-4,0},{15,1e-4,0},{10,5e-5,0}}',
    schedule = '{{40,2.5e-4,0},{10,1e-4,0},{10,5e-5,0}}',

    nThreads = 4,
    trainIters = 1000,
    testIters = 500,
    batchSize = 4,
    snapshot = 25,
    nGPU = 2,
    saveBest = 'true',
    subpixel_precision = 'true',
}


-- concatenate options fields to a string
local str_args = ''
for k, v in pairs(info) do
    str_args = str_args .. ('-%s %s '):format(k, v)
end

-- train network
exec_command(('th train.lua %s'):format(str_args))
