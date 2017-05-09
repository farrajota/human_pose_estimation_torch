
optim_method = 'adam'
nThreads = 4

info = {
    expID = 'hg-generic-ensemble-best',
    ensembleID = 'hg-generic-best',
    netType = 'hg-generic-ensemble',
    colourNorm = 'false',
    colourjit = 'true',
    centerjit = 0,
    pca = 'false',
    dropout = 0,
    spatialdropout = 0.2,
    critweights = 'none',
    
    scale = 0.3,
    rotate = 40,
    rotRate = 0,
    nStack = 8,
    nFeats = 256,
    schedule = '{{10,1e-3,0},{5,1e-4,0},{5,5e-5,0}}',
    batchSize = 8,
    snapshot = 0,
    nGPU = 2,
}

--
-- train ensemble flic
os.execute(('CUDA_VISIBLE_DEVICES=1,0 th train.lua -dataset flic -expID %s -netType %s -ensembleID %s'..
    ' -nGPU %d -optMethod %s -nThreads %d -colourNorm %s -colourjit %s'..
    ' -centerjit %d -dropout %.2f -spatialdropout %.2f -critweights %s -snapshot %d -schedule %s'..
    ' -rotRate %0.2f -nStack %d -nFeats %d -scale %0.2f -batchSize %d')
    :format(info.expID, info.netType, info.ensembleID, info.nGPU,
        optim_method, nThreads, info.colourNorm, 
        info.colourjit, info.centerjit, 
        info.dropout, info.spatialdropout, info.critweights,
        info.snapshot, info.schedule,
        info.rotRate, info.nStack, info.nFeats, info.scale, info.batchSize
    )
)