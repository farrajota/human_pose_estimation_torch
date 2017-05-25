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
  
--1 criterion weights
--1.1 - linear distribution
{expID = 'hg-generic'..nstacks..'-critweights-lin', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='lin', inputRes=256},
--1.2 - steeper linear distribution
{expID = 'hg-generic'..nstacks..'-critweights-steep', netType ='hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='steep', inputRes=256},
--1.3 - logaritmic distribution
{expID = 'hg-generic'..nstacks..'-critweights-log', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='log', inputRes=256},
--1.4 - exponential distribution
{expID = 'hg-generic'..nstacks..'-critweights-exp', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='exp', inputRes=256},

-- 2. spatial dropout middle (base model + spatial dropout in the middle of the residual block)
-- 2.1 - dropout rate = 0.1
{expID = 'hg-generic'..nstacks..'-spatialdropoutmiddle-0.1', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.1, critweights='false', inputRes=256},
-- 2.2 - dropout rate = 0.2
{expID = 'hg-generic'..nstacks..'-spatialdropoutmiddle-0.2', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.2, critweights='false', inputRes=256},
-- 2.3 - dropout rate = 0.3
{expID = 'hg-generic'..nstacks..'-spatialdropoutmiddle-0.3', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.3, critweights='false', inputRes=256},
-- 2.4 - dropout rate = 0.4
{expID = 'hg-generic'..nstacks..'-spatialdropoutmiddle-0.4', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.4, critweights='false', inputRes=256},
-- 2.5 - dropout rate = 0.5
{expID = 'hg-generic'..nstacks..'-spatialdropoutmiddle-0.5', netType = 'hg-genericv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0.5, critweights='false', inputRes=256},

-- 3. concat v1 (1 step concat before 1x1)
{expID = 'hg-generic'..nstacks..'-concatv1', netType = 'hg-generic-concatv1', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},

-- 4. concat v2 (1 step concat after 1x1)
{expID = 'hg-generic'..nstacks..'-concatv2', netType = 'hg-generic-concatv2', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256},

-- 5. higher rate of rotation
{expID = 'hg-generic'..nstacks..'-rotate', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256, rotRate=0.2},

-- 6. more hg stacks
--{expID = 'hg-generic'..10, netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256, nStack=10},

-- 7. denser feature maps
--{expID = 'hg-generic'..nstacks..'-nFeats', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256, nFeats=384},

-- 8. higher scaling factor
{expID = 'hg-generic'..nstacks..'-scale', netType = 'hg-generic', colourNorm='false', colourjit='false', centerjit=0, pca='false', dropout=0, spatialdropout=0, critweights='false', inputRes=256, scale=0.30},
}

local function TestArchScript(GPU, info)
    os.execute(('CUDA_VISIBLE_DEVICES=%d th train.lua -dataset flic -expID %s -netType %s'..
        ' -nGPU 1 -optMethod %s -nThreads %d -colourNorm %s -colourjit %s'..
        ' -centerjit %d -dropout %.2f -spatialdropout %.2f -critweights %s -snapshot %d -schedule %s'..
        ' -rotRate %0.2f -nStack %d -nFeats %d -scale %0.2f')
        :format(GPU, info.expID, info.netType,
            optim_method, nThreads,info.colourNorm, 
            info.colourjit, info.centerjit, 
            info.dropout, info.spatialdropout, info.critweights, snapshot, schedule,
            opt.rotRate or 0.6, opt.nStack or nstacks, opt.nFeats or 256, opt.scale or 0.25
            )
    )
end


for i=1+opt.n, #scripts, 2 do
    TestArchScript(opt.n,scripts[i])
end