--[[
    Train human pose predictor for FLIC and MPII datasets.
--]]

require 'torch'
require 'nn'
require 'optim'

-- load torchnet 
local tnt = require 'torchnet'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

-- load utility functions
local utils = paths.dofile('util/utils.lua')

-- Load data generator
paths.dofile('data.lua')

-- Load model
local net, criterion = paths.dofile('model.lua')

-- Load data augmentation functions
local imtransform = paths.dofile('transforms.lua')

-- Optimize networks memory usage
if opt.optimize then
  
end

-- Generate networks graph 
if opt.genGraph > 0 then
  
else
  
end

-- initializations
local tic = torch.tic()
local epoch = 0
local batchCounter = 1
local batchSize = opt.batchSize
local nfiles_train = dbimagenet.data.train.filename:size(1)
local nfiles_test = dbimagenet.data.test.filename:size(1)
-- some pre-computed ImageNet mean + std values
local meanstd = {
    mean = {0.48037518790839, 0.45039056120456, 0.39922636057037},
    std = {0.27660147027775, 0.26883440068399, 0.28014687231841},
}

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.data_type) end

print('==> Load network model')
local model = nn.Sequential()
local net = models[opt.model](opt.nGPU, #dbimagenet.data.train.classLabel)
if opt.data_type:match'torch.Cuda.*Tensor' then
   require 'cunn'
   if pcall(require, 'cudnn') then
     cudnn.convert(net, cudnn):cuda()
     cudnn.benchmark = true
     if opt.cudnn_deterministic then
        net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
     end
     
   end
   print(net)
   print('Network has', #net:findModules'cudnn.SpatialConvolution', 'cudnn convolutions')

   cast(net)
   local sample_input = torch.randn(8,3,net.imageCrop,net.imageCrop):cuda()
   if opt.optnet_optimize and opt.nGPU == 1 then
      optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end
end
model:add(net)
cast(model)


-- function that sets of dataset iterator:
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThreads,
      init    = function() 
                  require 'torch'
                  require 'torchnet' 
                  require 'image'
                end,
      closure = function()
        
         tic = torch.tic()
        
         -- load imagenet correct data set (train/test)
         local data = dbimagenet.data[mode]
         
         -- get random filename from a random class
         local ffi = require 'ffi'
         local function GetRandomFilename()
            local idx = torch.random(1, #data.classList)
            local randomIdx = torch.random(1, data.classList[idx].filenameIDList:size(1))
            local fidx = data.classList[idx].filenameIDList[randomIdx]
            local filename = ffi.string(data.filename[fidx]:data())
            return {filename, idx}
         end
         
         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
            list = torch.range(1, data.filename:size(1)):long(),
            load = function(idx)
                local data = GetRandomFilename()
                    return {
                        input = image.load(data[1],3,'float'),
                        target = torch.LongTensor{data[2]},
                    }
            end
          }
          
         return list_dataset
            :shuffle()
            :transform{
               input = tnt.transform.compose{
                  imtransform.Scale(256),
                  imtransform.RandomCrop(224),
                  imtransform.HorizontalFlip(0.5),
                  imtransform.ColorNormalize({mean = meanstd.mean, std = meanstd.std}),
                  imtransform.collectgarbage(),
               }
            }
            :batch(batchSize, 'include-last')
      end,
   }
end

-- set up training engine:
local engine = tnt.OptimEngine()
local criterion = cast(nn.CrossEntropyCriterion())
local meter  = tnt.AverageValueMeter()
local clerr  = tnt.ClassErrorMeter{topk = {1, 5}}
engine.hooks.onStartEpoch = function(state)
   print('\n***********************')
   print('Start Train epoch=' .. state.epoch+1)
   print('***********************')
   batchCounter = 1
   epoch = epoch + 1
   meter:reset()
   clerr:reset()
end
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
end

-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets = cast(torch.Tensor())
engine.hooks.onSample = function(state)
   inputs:resize(state.sample.input:size() ):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = inputs
   state.sample.target = targets
   state.sample.time = torch.toc(tic)
   tic=torch.tic()
end

engine.hooks.onBackward = function(state)
   if state.training then
      print(string.format('epoch[%d/%d][%d/%d][batch=%d] - loss: %2.4f; top-1 err: %2.2f; top-5 err: %2.2f; lr = %2.5f;  dataloading time: %0.5f; forward-backward time: %0.5f',epoch, state.maxepoch, batchCounter, math.floor(nfiles_train/batchSize), batchSize,
         meter:value(), clerr:value{k = 1}, clerr:value{k = 5}, state.config.learningRate, state.sample.time, torch.toc(tic)))
      batchCounter = batchCounter + 1
   end
end

engine.hooks.onForward = function(state)
   if not state.training then
      xlua.progress(batchCounter,  math.ceil(nfiles_test/batchSize))
      batchCounter = batchCounter + 1
   end
end

engine.hooks.onUpdate = function(state)
    tic = torch.tic()
end

engine.hooks.onEndEpoch = function(state)
  -- measure test loss and error:
  meter:reset()
  clerr:reset()
  batchCounter = 1
  print('\n***********************')
  print('Test network (epoch=' .. state.epoch .. ')')
  print('***********************')
  engine:test{
     network   = net,
     iterator  = getIterator('test'),
     criterion = criterion,
  }
  print(string.format('test loss: %2.4f;   top-1 error: %2.4f;   top-5 error: %2.4f',
     meter:value(), clerr:value{k = 1}, clerr:value{k = 5}))
    
   -- store model snapshot
   print('Saving model snapshot to: ' .. paths.concat(opt.path, opt.model .. '_model.t7'))
   if  state.epoch < state.maxepoch then
      saveDataParallel(paths.concat(opt.path, opt.model .. '_model.t7'), state.network)
    else
      saveDataParallel(paths.concat(opt.path, opt.model .. '_model.t7'), state.network:clearState())
   end
   print('Done.')
end

-- train the model:
print('==> Train network model')
engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = {
     learningRate = 1e-2,
     momentum = 0.9,
     weightDecay = 5e-4,
   },
   maxepoch = opt.nEpoch
}

print('==> Script complete.')