--[[
    Load model into memory.
]]

require 'nn'

-- Continuing an experiment where it left off
local model
opt.iniEpoch = 1
if opt.continue or opt.branch ~= 'none' then
    local optimState = torch.load(opt.save .. '/optimState.t7')
    local prevModel = opt.save .. '/model_' .. optimState.epoch .. '.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)
    opt.iniEpoch =  optimState.epoch
    epoch = optimState.epoch

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
    model = models_list[opt.netType](opt)
end

-- define criterion
local criterion
if string.match('MSE', string.upper(opt.crit)) then
  criterion = nn.MSECriterion()
elseif string.match('smoothl1', string.lower(opt.crit)) then
  criterion = nn.SmoothL1Criterion()
end

-- convert to GPU or CPU
if opt.GPU > 1 then
   print('Running on GPU: [' .. opt.nGPU .. ']')
   require 'cutorch'
   require 'cunn'
   model:cuda()
   criterion:cuda()
  
   -- check if cudnn is available
   if pcall(require, 'cudnn') then
     cudnn.convert(net, cudnn):cuda()
     cudnn.benchmark = true
     if opt.cudnn_deterministic then
        model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
     end
     print('Network has', #net:findModules'cudnn.SpatialConvolution', 'cudnn convolutions')
   end
else
  print('Running on CPU')
  model:float()
  criterion:float()
end

----------------------------------

return model, criterion