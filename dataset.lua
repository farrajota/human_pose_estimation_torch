--[[
    Load dataset
]]


local dbclt = require 'dbcollection'


-------------------------------------------------------------------------------
-- Load Dataset
-------------------------------------------------------------------------------

function loadDataset(dataset)
--
-- Loads a dataset's metadata into memory.
-- 

    local data = {train = {}, val = {}}
    local dataset = dataset or opt.dataset
    
    -- train/test scripts
    local category = 'full'
    if dataset == 'mpii' then
        category = 'singletrainval'
    elseif dataset == 'lsp' then
        category = 'extended'
    elseif dataset == 'mscoco' or dataset == 'coco' then
        category = 'filtered'
    end
    
    -- load dataset
    local dbdataset 
    if dataset ~= 'mpii+lsp' then
        dbdataset = dbclt.get{name=dataset, task = 'keypoints', category=category}
    end
    
    -- load corresponding set
    if dataset == 'mpii' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.val
        data.test = dbdataset.data.test
        
    elseif dataset == 'flic' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.test
        data.test = dbdataset.data.test
        
    elseif dataset == 'lsp' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.test
        data.test = dbdataset.data.test
        
    elseif dataset == 'mscoco' or dataset == 'coco' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.val
        data.test = dbdataset.data.test
    
    elseif dataset == 'mpii+lsp' then
      
        local dbdatasetMPII = dbclt.get{name='mpii', task = 'keypoints', category='full'}
        local dbdatasetLSP = dbclt.get{name='lsp', task = 'keypoints', category='full'}
        local dbdatasetLSPe = dbclt.get{name='lsp', task = 'keypoints', category='extended'}
        
        local nObjects = dbdatasetMPII.data.train.object:size(1) + dbdatasetLSP.data.train.object:size(1) + dbdatasetLSPe.data.train.object:size(1)
        
        data.train = { 
            object = torch.range(1,nObjects), 
            data = { dbdatasetMPII.data.train, dbdatasetLSP.data.train, dbdatasetLSPe.data.train },
            isTrain = true
        }
        data.val = dbdatasetLSPe.data.test
        data.test = dbdatasetLSPe.data.test
    
    else
        error('Invalid dataset: ' .. dataset..'. Please use one of the following datasets: mpii, flic, lsp, mscoco.')
    end
    
    return data
end

