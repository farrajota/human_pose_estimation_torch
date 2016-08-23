--[[
    Utility functions.
]]


local function MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

local function FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

local function DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

local function makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      
      if pcall(require,'cudnn') then
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark
        local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            require 'scalegrad'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
        dpt.gradInput = nil
      else
        local dpt = nn.DataParallelTable(1, true, true)
           :add(model, gpus)
           :threads(function()
              local cudnn = require 'cunn'
           end)
        dpt.gradInput = nil
      end

      model = dpt:cuda()
   end
   return model
end

-----------------------------------------

return {
  makeDataParallelTable = makeDataParallelTable,
 }
