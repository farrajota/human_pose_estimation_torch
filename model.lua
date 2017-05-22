--[[
    Load model network into memory.
]]


--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------

-- Continuing an experiment where it left off
local model
opt.iniEpoch = 1
if opt.continue or opt.branch ~= 'none' then
    local prevModel
    if paths.filep(opt.save .. '/optim.t7') then
        prevModel = opt.save .. '/model.t7'
    else
        prevModel = opt.save .. '/model_' .. epoch .. '.t7'
    end

    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)
    opt.iniEpoch = epoch

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    -- load models
    local models_list = paths.dofile('models/init.lua')
    assert(models_list[opt.netType], 'Undefined model architecture: ' .. opt.netType)
    model = models_list[opt.netType]()
end


--------------------------------------------------------------------------------
-- Define criterion
--------------------------------------------------------------------------------

local criterion
if opt.nOutputs > 0 then
    criterion = nn.ParallelCriterion()

    local limits = {1,2}
    local w
    if opt.critweights == 'linear' then
        w = torch.linspace(limits[1], limits[2], opt.nOutputs)
    elseif opt.critweights == 'steep' then
        w = torch.linspace(limits[1], limits[2]*5, opt.nOutputs)
    elseif opt.critweights == 'log' then
        w = torch.linspace(limits[1], limits[2], opt.nOutputs):mul(2):log()
    elseif opt.critweights == 'exp' then
        w = torch.linspace(limits[1], limits[2], opt.nOutputs):exp()
    else
        w = torch.IntTensor(opt.nOutputs):fill(1)
    end

    for i=1, opt.nOutputs do
        local weight = w[i]
        if string.match('MSE', string.upper(opt.crit)) then
            criterion:add(nn.MSECriterion(), weight)
        elseif string.match('smoothl1', string.lower(opt.crit)) then
            criterion:add(nn.SmoothL1Criterion(), weight)
        end
    end
else
    error('Must have more than 1 stack: '.. opt.nOutputs)
    --if string.match('MSE', string.upper(opt.crit)) then
    --    criterion = nn.MSECriterion()
    --elseif string.match('smoothl1', string.lower(opt.crit)) then
    --    criterion = nn.SmoothL1Criterion()
    --end
end


--------------------------------------------------------------------------------
-- Convert to GPU or CPU
--------------------------------------------------------------------------------

if opt.GPU >= 1 then
    print('Running on GPU: [' .. opt.nGPU .. ']')
    require 'cutorch'
    require 'cunn'
    model:cuda()
    criterion:cuda()

   -- require cudnn if available
    if pcall(require, 'cudnn') then
        cudnn.convert(model, cudnn):cuda()
        cudnn.benchmark = true
        if opt.cudnn_deterministic then
            model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
        end
        print('Network has', #model:findModules'cudnn.SpatialConvolution', 'cudnn convolutions')
    end
    opt.dataType = 'torch.CudaTensor'
else
    print('Running on CPU')
    model:float()
    criterion:float()
    opt.dataType = 'torch.FloatTensor'
end


--------------------------------------------------------------------------------
-- Config network to use multiple GPUs
--------------------------------------------------------------------------------

local modelOut = nn.Sequential()

-- Generate networks graph
if opt.genGraph > 0 and epoch == 1 then
    graph.dot(model.fg, 'pose network', paths.concat(opt.save, 'network_graph'))
    local sys = require 'sys'
    if #sys.execute('command -v inkscape') > 0 then
        os.execute(('inkscape -z -e %s  -h 30000 %s'):format(paths.concat(opt.save, 'network_graph.png'),  paths.concat(opt.save, 'network_graph.svg')))
    end
end


-- Use multiple gpus
if opt.GPU >= 1 and opt.nGPU > 1 then
    if torch.type(model) == 'nn.DataParallelTable' then
        modelOut:add(utils.loadDataParallel(model, opt.nGPU))
    else
        modelOut:add(utils.makeDataParallelTable(model, opt.nGPU))
    end
else
    modelOut:add(model)
end

local function cast(x) return x:type(opt.data_type) end

cast(modelOut)


--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return modelOut, criterion