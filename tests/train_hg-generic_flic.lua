--[[
    
]]

optim_method = 'adam'
nThreads = 4


local info = {
    expID = 'hg-generic8-v2',
    netType = 'hg-generic',
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
    nStack = 8,
    nFeats = 256,
    schedule = '{{50,2.5e-4,0},{15,1e-4,0},{10,5e-5,0}}',
    --schedule = '{{40,2.5e-4,0},{10,1e-4,0},{10,5e-5,0}}',
    batchSize = 4,
    snapshot = 25,
    nGPU = 1,
    saveBest = 'false'
}


-- train model flic
os.execute(('CUDA_VISIBLE_DEVICES=1 th train.lua -dataset flic -expID %s -netType %s'..
    ' -nGPU %d -optMethod %s -nThreads %d -colourNorm %s -colourjit %s'..
    ' -centerjit %d -dropout %.2f -spatialdropout %.2f -critweights %s -snapshot %d -schedule %s'..
    ' -rotRate %0.2f -nStack %d -nFeats %d -scale %0.2f -batchSize %d -saveBest %s')
    :format(info.expID, info.netType, info.nGPU,
        optim_method, nThreads, info.colourNorm, 
        info.colourjit, info.centerjit, 
        info.dropout, info.spatialdropout, info.critweights,
        info.snapshot, info.schedule,
        info.rotRate, info.nStack, info.nFeats, info.scale, info.batchSize, info.saveBest
    )
)

