--[[
    Train human pose predictor for FLIC and MPII datasets.
--]]

require 'torch'
require 'paths'
require 'string'


--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

paths.dofile('configs.lua')
local lopt = opt

print('\n*******')
print(opt.optMethod)
print('*******\n')

-- load torchnet package
local tnt = require 'torchnet'

-- set local vars for the number of total files int the train and test sets

local nBatchesTrain = opt.trainIters
local nBatchesTest = opt.validIters


-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

-- load dataset
paths.dofile('dataset.lua')
local dataset = loadDataset()
local nItersTrain = dataset.train.object:size(1)
local nItersTest = dataset.val.object:size(1)

--[[
paths.dofile('data_new.lua')
local data=loadDataset('train')
a,b = loadData(data, 1, 'train')
--a,b = loadData('train', 150, 2)
aqui=1
--]]

--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThreads,
      init    = function(threadid) 
                  require 'torch'
                  require 'torchnet'
                  opt = lopt
                  paths.dofile('data_new.lua')
                  torch.manualSeed(threadid+opt.manualSeed)
                end,
      closure = function()
         
         -- setup data
         local data = dataset[mode]
         
         -- number of iterations
         local nIters = data.object:size(1)
         
         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
            list = torch.range(1, nIters):long(),
            load = function(idx)
                local input, label = loadData(data, idx, mode)
                    return {
                        input = input,
                        target = label,
                    }
            end
          }
          
         return list_dataset
            :shuffle()
            :batch(opt.batchSize, 'include-last')
      end,
   }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local meters = {
   train_err = tnt.AverageValueMeter(),
   train_accu = tnt.AverageValueMeter(),
   valid_err = tnt.AverageValueMeter(),
   valid_accu = tnt.AverageValueMeter(),
}

function meters:reset()
   self.train_err:reset()
   self.train_accu:reset()
   self.valid_err:reset()
   self.valid_accu:reset()
end

local loggers = {
   valid = optim.Logger(paths.concat(opt.save,'valid.log')),
   train = optim.Logger(paths.concat(opt.save,'train.log')),
   full_train = optim.Logger(paths.concat(opt.save,'full_train.log')),
}

loggers.valid:setNames{'Valid Loss', 'Valid acc.'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss', 'Train accuracy'}

loggers.valid.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false


-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
   if state.training then
      if opt.iniEpoch>1 then 
         state.epoch = math.max(opt.iniEpoch, state.epoch)
      end
   end
end

engine.hooks.onStartEpoch = function(state)
    print('\n**********************************************')
    print(('Starting Train epoch %d/%d'):format(state.epoch+1, state.maxepoch))
    print('**********************************************')
    state.config = optimStateFn(state.epoch+1)
end


engine.hooks.onForwardCriterion = function(state)
    if state.training then
        xlua.progress((state.t+1)*opt.batchSize, nItersTrain)
        
        -- compute the PCK accuracy of the networks (last) output heatmap with the ground-truth heatmap
        local acc = accuracy(state.network.output, state.sample.target)
        
        meters.train_err:add(state.criterion.output)
        meters.train_accu:add(acc)
        loggers.full_train:add{state.criterion.output, acc}
    else
        xlua.progress(state.t*opt.batchSize, nItersTest)
        
        -- compute the PCK accuracy of the networks (last) output heatmap with the ground-truth heatmap
        local acc = accuracy(state.network.output, state.sample.target)
        
        meters.valid_err:add(state.criterion.output)
        meters.valid_accu:add(acc)
    end
end

-- copy sample to GPU buffer:
local samples = {cast(torch.Tensor()), cast(torch.Tensor())}
local function ReplicateTensor2Table(tensor, N)
    local data = {}
    if N > 1 then
        for i=1, N do table.insert(data, tensor) end
        return data
    else
        return tensor
    end
end


engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    samples[1]:resize(state.sample.input:size() ):copy(state.sample.input)
    samples[2]:resize(state.sample.target:size() ):copy(state.sample.target)
    
    state.sample.input  = samples[1]
    state.sample.target = ReplicateTensor2Table(samples[2], opt.nOutputs)
end


engine.hooks.onEndEpoch = function(state)
   print(('Train Loss: %0.5f; Acc: %0.5f'):format(meters.train_err:value(),  meters.train_accu:value()))
   -- measure test loss and error:
   local tr_loss = meters.train_err:value()
   local tr_accuracy = meters.train_accu:value()
   loggers.train:add{tr_loss, tr_accuracy}
   meters:reset()
   state.t = 0
   
   print('\n**********************************************')
   print(('Test network (epoch = %d/%d)'):format(state.epoch, state.maxepoch))
   print('**********************************************')
   engine:test{
      network   = model,
      iterator  = getIterator('val'),
      criterion = criterion,
   }
   local vl_loss = meters.valid_err:value()
   local vl_accuracy = meters.valid_accu:value()
   loggers.valid:add{vl_loss, vl_accuracy}
   print(('Validation Loss: %0.5f; Acc: %0.5f'):format(meters.valid_err:value(),  meters.valid_accu:value()))
   
   -- store model
   storeModel(state.network.modules[1], state.config, state.epoch, opt)
   state.t = 0
end


--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------
print('==> Train network model')
engine:train{
   network   = model,
   iterator  = getIterator('train'),
   criterion = criterion,
   optimMethod = optim[opt.optMethod],
   config = optimState,
   maxepoch = opt.nEpochs
}


--------------------------------------------------------------------------------
-- Save model
--------------------------------------------------------------------------------
print('==> Saving final model to disk: ' .. paths.concat(opt.save,'final_model.t7'))
local utils = paths.dofile('util/utils.lua')
utils.saveDataParallel(paths.concat(opt.save,'final_model.t7'), model.modules[1]:clearState())
torch.save(paths.concat(opt.save,'final_optimState.t7'), optimStateFn(opt,nEpochs))
loggers.valid:style{'+-', '+-'}; loggers.valid:plot()
loggers.train:style{'+-', '+-'}; loggers.train:plot()
loggers.full_train:style{'-', '-'}; loggers.full_train:plot()

print('==> Script complete.')