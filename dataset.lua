--[[
    Load dataset
]]


local dbclt = require 'dbcollection'


-------------------------------------------------------------------------------
-- Load Dataset
-------------------------------------------------------------------------------

function loadDataset()

    local data = {train = {}, val = {}}
    
    -- train/test scripts
    local category = 'full'
    if opt.dataset == 'mpii' then
        category = 'singletrainval'
    elseif opt.dataset == 'lsp' then
        category = 'extended'
    elseif opt.dataset == 'mscoco' or opt.dataset == 'coco' then
        category = 'filtered'
    end
    
    -- load dataset
    local dbdataset = dbclt.get{name=opt.dataset, task = 'keypoints', category=category}
    
    -- load corresponding set
    if opt.dataset == 'mpii' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.val
        data.test = dbdataset.data.test
        
    elseif opt.dataset == 'flic' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.test
        data.test = dbdataset.data.test
        
    elseif opt.dataset == 'lsp' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.test
        data.test = dbdataset.data.test
        
    elseif opt.dataset == 'mscoco' or opt.dataset == 'coco' then
        data.train = dbdataset.data.train
        data.val = dbdataset.data.val
        data.test = dbdataset.data.test
        
    else
        error('Invalid dataset: ' .. opt.dataset..'. Please use one of the following datasets: mpii, flic, lsp, mscoco.')
    end
    
    return data
end

