--[[
    Logger functions for easier logging data management.
]]


local utils = paths.dofile('utils.lua')

------------------------------------------------------------------------------------------------------------

local function reOptimizeNetwork()
   local optnet = require 'optnet'
   local sample_input = torch.randn(math.max(1,math.floor(opt.batchSize/4)), 3, opt.cropSize, opt.cropSize):float()
   if opt.GPU>=1 then sample_input:cuda() end
   optnet.optimizeMemory(model, sample_input, {inplace = false, mode = 'training'})
end

------------------------------------------------------------------------------------------------------------

local function store(model, optimState, epoch, flag, flag_optimize)
   local flag_optimize = flag_optimize or false
   if flag then 
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model_' .. epoch ..'.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model.modules[1]:clearState())
      --torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:clearState())
      
      torch.save(paths.concat(opt.save,'optim_' .. epoch ..'.t7'), optimState)
      torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)
      
      -- re-optimize network
      if flag_optimize then
        reOptimizeNetwork()
      end
   else
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model.t7'), model.modules[1]:clearState())
      --torch.save(paths.concat(opt.save, 'model.t7'), model:clearState())
      
      torch.save(paths.concat(opt.save,'optim.t7'), optimState)
      torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)
      
      -- re-optimize network
      if flag_optimize then
        reOptimizeNetwork()
      end
   end
end

------------------------------------------------------------------------------------------------------------  

function storeModel(model, optimState, epoch, opt)
   -- store model snapshot
   if opt.snapshot > 0 then
     if epoch%opt.snapshot == 0 then 
        store(model, optimState, epoch, true)
     end
   
   elseif opt.snapshot < 0 then
     if epoch%math.abs(opt.snapshot) == 0 then 
        store(model, optimState, epoch, false)
     end
   else 
      -- save only at the last epoch
      if epoch == opt.nEpochs then
        store(model, optimState, epoch, false)
      end
   end
end