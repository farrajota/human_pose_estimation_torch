--[[
    Logger functions for easier logging data management.
]]


local utils = paths.dofile('utils.lua')

------------------------------------

local function reOptimizeNetwork()
   local optnet = require 'optnet'
   local sample_input = torch.randn(math.max(1,math.floor(opt.batchSize/4)), 3, opt.cropSize, opt.cropSize):float()
   if opt.GPU>=1 then sample_input:cuda() end
   optnet.optimizeMemory(model, sample_input, {inplace = false, mode = 'training'})
end

------------------------------------

local function storeModel(model, optimState, epoch, flag)
   if flag then 
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model_' .. epoch ..'.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:clearState())
      
      torch.save(paths.concat(opt.save,'optim_' .. epoch ..'.t7'), optimstate)
      torch.save(paths.concat(opt.save,'meters_' .. epoch ..'.t7'), meters)
      
      -- re-optimize network
      if opt.optimize then
        reOptimizeNetwork()
      end
   else
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model.t7'), model:clearState())
      
      torch.save(paths.concat(opt.save,'optim.t7'), optimstate)
      torch.save(paths.concat(opt.save,'meters.t7'),meters)
      
      -- re-optimize network
      if opt.optimize then
        reOptimizeNetwork()
      end
   end
end

------------------------------------  

local function logging(model, optimState, loggers, epoch)

   loggers.test:style{'+-','+-','+-'}
   loggers.test:plot()

   loggers.train:style{'+-','+-'}
   loggers.train:plot()

   loggers.full_train:style{'+-'}
   loggers.full_train:plot()

   -- store model snapshot
   if opt.snapshot > 0 then
     if epoch%opt.snapshot == 0 then 
        storeModel(model, optimState, epoch, true)
     end
   
   elseif opt.snapshot < 0 then
     if epoch%math.abs(opt.snapshot) == 0 then 
        storeModel(model, optimState, epoch, false)
     end
   else 
      -- save only at the last epoch
      if epoch == opt.nEpochs then
        storeModel(model, optimState, epoch, false)
      end
   end

end

------------------------------------

return logging
