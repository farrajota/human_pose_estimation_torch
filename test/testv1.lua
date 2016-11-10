--[[
    Test several architectures script. (version 1)
]]


local cmd = torch.CmdLine()
cmd:text()
cmd:text(' ---------- General options for architect test v1 ------------------------------------')
cmd:text()
cmd:option('-n',  0, '0- even | 1- odd')
cmd:text()
local opt = cmd:parse(arg or {})

--------------------------------------------------------------------------------
-- Initializations
--------------------------------------------------------------------------------

local nThreads = 2
local optim_method = 'adam'
local batchSize = 4
local nstacks = 8
local snapshot = 0
local schedule = "{{30,2.5e-4,0},{5,1e-4,0},{5,5e-5,0}}"

local scripts = {
    -- 1 (base model)
    {expID = 'hg-generic'..nstacks, netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    -- 2 (base model + mean/std norm)
    {expID = 'hg-generic'..nstacks..'-meanstd', netType = 'hg-generic', colourNorm='true', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    -- 3 (base model + different colour jitter method)
    {expID = 'hg-generic'..nstacks..'-colourjit', netType = 'hg-generic', colourNorm='false', colourjit='true', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    -- 4 (base model + higher input resolution)
    {expID = 'hg-generic'..nstacks..'-highres', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=512},
    -- 5 (base model + center coords jit (+/- 5px))
    {expID = 'hg-generic'..nstacks..'-centerjit', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=5, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    -- 6 (base model + dropout before the residual block)
    {expID = 'hg-generic'..nstacks..'-dropoutbefore', netType = 'hg-genericv3', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0.2, spatialdropout=0, critweights='false', inputRes=256},
    -- 7 (base model + dropout in the middle of the residual block)
    {expID = 'hg-generic'..nstacks..'-dropoutmiddle', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0.2, spatialdropout=0, critweights='false', inputRes=256},
    -- 8 (base model + spatial dropout before the residual block)
    {expID = 'hg-generic'..nstacks..'-spatialdropoutbefore', netType = 'hg-genericv3', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.2, critweights='false', inputRes=256},
    --9 (base model + spatial dropout in the middle of the residual block)
    {expID = 'hg-generic'..nstacks..'-spatialdropoutmiddle', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.2, critweights='false', inputRes=256},
    --10 (base model + criterion with different weights (higer on the later ones) )
    {expID = 'hg-generic'..nstacks..'-critweights', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='true', inputRes=256},
    --11 (1 step concat before 1x1)
    {expID = 'hg-generic'..nstacks..'-concatv1', netType = 'hg-generic-concatv1', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    --12 (1 step concat after 1x1)
    {expID = 'hg-generic'..nstacks..'-concatv2', netType = 'hg-generic-concatv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    --13 (concat all prior outputs prior to the current one )
    {expID = 'hg-generic'..nstacks..'-concatdense', netType = 'hg-generic-concatdense', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    --14 (bypass one output with the previous one (caddtable))
    {expID = 'hg-generic'..nstacks..'-bypass', netType = 'hg-generic-bypass', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
    --14 (bypass one output with all the previous ones (caddtable))
    {expID = 'hg-generic'..nstacks..'-bypassdense', netType = 'hg-generic-bypassdense', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},
}

local function TestArchScript(GPU, info)
    os.execute(('CUDA_VISIBLE_DEVICES=%d th train.lua -dataset flic -expID %s -netType %s'..
        ' -nGPU 1 -optMethod %s -nThreads %d -colourNorm %s -colourjit %s'..
        ' -centerjit %d -pca %s -dropout %.2f -spatialdropout %.2f -critweights %s -snapshot %d -schedule %s')
        :format(GPU, info.expID, info.netType,
            optim_method, nThreads,info.colourNorm, 
            info.colourjit, info.centerjit, info.pca, 
            info.dropout, info.spatialdropout, info.critweights, snapshot, schedule)
    )
end


for i=1+opt.n, #scripts, 2 do
    TestArchScript(0,scripts[i])
end