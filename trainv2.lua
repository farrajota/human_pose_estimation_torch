--[[
    Train human pose predictor for FLIC and MPII datasets.
--]]

require 'torch'
require 'paths'
require 'string'

-- load configs (data, model, criterion, optimState)
paths.dofile('configs.lua')
local lopt = opt

-- load torchnet package
local tnt = require 'torchnet'

-- set local vars for the number of total files int the train and test sets
local lloadData = loaddata
local dbimagenet = dbimagenet
local nBatchesTrain = math.floor(dbimagenet.data.train.filename:size(1)/opt.batchSize)
local nBatchesTest = math.ceil(dbimagenet.data.test.filename:size(1)/opt.batchSize)

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

-- data generator
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThreads,
      init    = function(threadid) 
                  require 'torch'
                  require 'torchnet' 
                  t = paths.dofile('transforms.lua')
                  opt = lopt
                  torch.manualSeed(threadid+opt.manualSeed)
                end,
      closure = function()
         
         local nIters = (mode == 'train' and opt.trainIters) or (mode == 'valid' and opt.validIters)
         local batchSize = (mode == 'train' and opt.trainBatch) or (mode == 'valid' and opt.validBatch)
         
         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
            list = torch.range(1, nIters):long(),
            load = function(idx)
                local input, label = lloadData(mode, idx)
                    return {
                        input = input,
                        target = label,
                    }
            end
          }
          
         return list_dataset
            :shuffle()
            :transform{
              input = mode == 'train' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(opt.imageSize),
                        t.RandomCrop(opt.cropSize),
                        t.ColorNormalize({mean = opt.meanstd.mean, std = opt.meanstd.std}),
                        t.HorizontalFlip(0.5),
                     }
                  or mode == 'valid' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(opt.imageSize),
                        t.CenterCrop(opt.cropSize),
                        t.ColorNormalize({mean = opt.meanstd.mean, std = opt.meanstd.std}),
                     }
            }
            :batch(batchSize, 'include-last')
      end,
   }
end

local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local meters = {
   conf = tnt.ConfusionMeter{k = opt.nClasses},
   val = tnt.AverageValueMeter(),
   train = tnt.AverageValueMeter(),
   train_clerr = tnt.ClassErrorMeter{topk = {1, 5},accuracy=true},
   clerr = tnt.ClassErrorMeter{topk = {1,5},accuracy=true},
   ap = tnt.APMeter(),
}

function meters:reset()
   self.conf:reset()
   self.val:reset()
   self.train:reset()
   self.train_clerr:reset()
   self.clerr:reset()
   self.ap:reset()
end

local loggers = {
   test = optim.Logger(paths.concat(opt.save,'test.log')),
   train = optim.Logger(paths.concat(opt.save,'train.log')),
   full_train = optim.Logger(paths.concat(opt.save,'full_train.log')),
}

loggers.test:setNames{'Test Loss', 'Test acc.', 'Test mAP'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false
------------

-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStartEpoch = function(state)
   print('\n***********************')
   print('Start Train epoch=' .. state.epoch+1)
   print('***********************')
   timers.epochTimer:reset()
   state.config = optimStateFn(state.epoch+1)
end

engine.hooks.onForwardCriterion = function(state)
   if state.training then
      print(string.format('epoch[%d/%d][%d/%d][batch=%d] - loss: %2.4f; top-1 err: %2.2f; top-5 err: %2.2f; lr = %2.5f;  DataLoadingTime: %0.5f; forward-backward time: %0.5f', state.epoch+1, state.maxepoch, state.t+1, nBatchesTrain, opt.batchSize,
         meters.train:value(), meters.train_clerr:value{k = 1}, meters.train_clerr:value{k = 5}, state.config.learningRate, timers.dataTimer:time().real, timers.batchTimer:time().real))
      timers.batchTimer:reset()
      meters.train:add(state.criterion.output)
      meters.train_clerr:add(state.network.output,state.sample.target)
      loggers.full_train:add{state.criterion.output}
   else
      meters.conf:add(state.network.output,state.sample.target)
      meters.clerr:add(state.network.output,state.sample.target)
      meters.val:add(state.criterion.output)
      local tar = torch.ByteTensor(#state.network.output):fill(0)
      for k=1,state.sample.target:size(1) do
         local id = state.sample.target[k]:squeeze()
         tar[k][id]=1
      end
      meters.ap:add(state.network.output,tar)
   end
end

-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets = cast(torch.Tensor())
engine.hooks.onSample = function(state)
   inputs:resize(state.sample.input:size() ):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = inputs
   state.sample.target = targets
   if opt.task == 'pose-int' then
      local newTargets = {}
      for i=1, opt.nStacks do 
         table.insert(newTargets, state.sample.target) 
      end
      state.sample.target = newTargets
   end
   timers.dataTimer:stop()
end

engine.hooks.onForward = function(state)
   if not state.training then
      xlua.progress(state.t, nBatchesTest)
   end
end

engine.hooks.onUpdate = function(state)
    timers.dataTimer:reset()
    timers.dataTimer:resume()
end

engine.hooks.onEndEpoch = function(state)
   print("Epoch Train Loss:" ,meters.train:value(),"Total Epoch time: ",timers.epochTimer:time().real)
   -- measure test loss and error:
   loggers.train:add{meters.train:value(),meters.train_clerr:value()[1]}
   meters:reset()
   state.t = 0
   print('\n***********************')
   print('Test network (epoch=' .. state.epoch .. ')')
   print('***********************')
   engine:test{
      network   = model,
      iterator  = getIterator('valid'),
      criterion = criterion,
   }

   loggers.test:add{meters.val:value(),meters.clerr:value()[1],meters.ap:value():mean()}
   print("Validation Loss" , meters.val:value())
   print("Accuracy: Top 1%", meters.clerr:value{k = 1})
   print("Accuracy: Top 5%", meters.clerr:value{k = 5})
   print("mean AP:",meters.ap:value():mean())
   log(state.network, state.config, meters, loggers, state.epoch)
   print("Testing Finished")
   timers.epochTimer:reset()
   state.t = 0
end

-- train the model:
print('==> Train network model')
engine:train{
   network   = model,
   iterator  = getIterator('train'),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = {
     learningRate = opt.LR,
     momentum = opt.momentum,
     weightDecay = opt.weightDecay,
   },
   maxepoch = opt.nEpoch
}

print('==> Script complete.')