--[[
    Load dataset
]]


local dbclt = require 'dbcollection'


-------------------------------------------------------------------------------
-- Load Dataset
-------------------------------------------------------------------------------

function loadDataset(dataset)

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
    local dbdataset = dbclt.get{name=dataset, task = 'keypoints', category=category}
    
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
        
    else
        error('Invalid dataset: ' .. dataset..'. Please use one of the following datasets: mpii, flic, lsp, mscoco.')
    end
    
    return data
end

