--[[
    Loads necessary libraries and files for the train script.
]]

-------------------------------------------------------------------------------
-- Load necessary libraries and files
-------------------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/Logger.lua')
paths.dofile('util/store.lua')

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
paths.dofile('projectdir.lua')


-------------------------------------------------------------------------------
-- Process command line options
-------------------------------------------------------------------------------

if not opt then

  local opts = paths.dofile('opts.lua')
  opt = opts.parse(arg)

  print('Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  if opt.GPU >= 1 then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.GPU)
  end

  if opt.branch ~= 'none' or opt.continue then
      -- Continuing training from a prior experiment
      -- Figure out which new options have been set
      local setOpts = {}
      for i = 1,#arg do
          if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
      end

      -- Where to load the previous options/model from
      if opt.branch ~= 'none' then opt.load = opt.expDir .. '/' .. opt.branch
      else opt.load = opt.expDir .. '/' .. opt.expID end

      -- Keep previous options, except those that were manually set
      local opt_ = opt
      opt = torch.load(opt_.load .. '/options.t7')
      opt.save = opt_.save
      opt.load = opt_.load
      opt.continue = opt_.continue
      for i = 1,#setOpts do opt[setOpts[i]] = opt_[setOpts[i]] end

      epoch = opt.lastEpoch + 1
      
      -- If there's a previous optimState, load that too
      if paths.filep(opt.load .. '/optimState.t7') then
          optimState = torch.load(opt.load .. '/optimState.t7')
          optimState.learningRate = opt.LR
      end

  else epoch = 1 end
  opt.epochNumber = epoch

  -- Training hyperparameters
  -- (Some of these aren't relevant for rmsprop which is the optimization we use)
  if not optimState then
      optimState = {
          learningRate = opt.LR,
          learningRateDecay = opt.LRdecay,
          momentum = opt.momentum,
          dampening = 0.0,
          weightDecay = opt.weightDecay
      }
  end
  
  -- define optim state function ()
  optimStateFn = function(epoch) return optimState end

  -- Random number seed
  if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
  else torch.seed() end                           

  -- Save options to experiment directory
  torch.save(opt.save .. '/options.t7', opt)

end


-------------------------------------------------------------------------------
-- Load model + criterion
-------------------------------------------------------------------------------

if not outputDim then
  if opt.dataset == 'mpii' then
    outputDim = {16, opt.outputRes, opt.outputRes}
  elseif opt.dataset == 'flic' then
    outputDim = {11, opt.outputRes, opt.outputRes}
  else
    error('Undefined dataset name: ' .. opt.dataset)
  end
end

if opt.netType == 'hg-stacked' then
   if opt.task == 'pose-int' then
      local newDim = {}
      newDim[1] = outputDim
      newDim[2] = outputDim
      outputDim = newDim
      opt.nOutputs = 2
   else
      opt.nOutputs = 1
   end
else
  local newDim = {}
  for i=1, opt.nStack do
    table.insert(newDim, outputDim)
  end
  outputDim = newDim
  opt.nOutputs = opt.nStack
end

local net, crit = paths.dofile('model.lua') -- Load model
criterion = crit
model = nn.Sequential()


--------------------------------------------------------------------------------
-- Optimize networks memory usage
--------------------------------------------------------------------------------

-- optimize network's memory allocations
if opt.optimize then
   -- for memory optimizations and graph generation
   print('Optimize (reduce) network\'s memory usage...')
   local optnet = require 'optnet'
  
   local sample_input = torch.randn(2,3,opt.inputRes,opt.inputRes):float()
   if opt.GPU>=1 then sample_input=sample_input:cuda() end
   optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   print('Done.')
end

-- Generate networks graph 
if opt.genGraph > 0 then
  graph.dot(net.fg, 'pose network', paths.concat(opt.save, 'network_graph'))
  local sys = require 'sys'
  if #sys.execute('command -v inkscape') > 0 then
    os.execute(('inkscape -z -e %s  -h 30000 %s'):format(paths.concat(opt.save, 'network_graph.png'),  paths.concat(opt.save, 'network_graph.svg')))
  end
end

-- Use multiple gpus
if opt.GPU >= 1 and opt.nGPU > 1 then
  local utils = paths.dofile('util/utils.lua')
  model:add(utils.makeDataParallelTable(net, opt.nGPU)) -- defined in util.lua
else
  model:add(net)
end

local function cast(x) return x:type(opt.data_type) end

cast(model)
