--[[
    Loads necessary libraries and files for the benchmark script.
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
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/Logger.lua')
paths.dofile('util/store.lua')
utils = paths.dofile('util/utils.lua')

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
paths.dofile('projectdir.lua')

--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Pose Benchmark (PCK(h) evaluation) options: ')
cmd:text()
cmd:text(' ---------- General options --------------------------------------')
cmd:text()
cmd:option('-expID',       'hg-generic8', 'Experiment ID')
cmd:option('-dataset',     'flic', 'Dataset choice: mpii | flic')
cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
cmd:option('-reprocess',   false,  'Utilize existing predictions from the model\'s folder.')
cmd:option('-manualSeed',      2, 'Manually set RNG seed')
cmd:option('-GPU',             1, 'Default preferred GPU (-1 = use CPU)')
cmd:option('-nGPU',            1, 'Number of GPUs to use by default')
cmd:option('-nThreads',        2, 'Number of data loading threads')
cmd:option('-threshold',     0.2, 'PCKh threshold (default 0.5)')
cmd:option('-predictions',     0, 'Generate a predictions file (0-false | 1-true)')
cmd:text()
cmd:text(' ---------- Model options --------------------------------------')
cmd:text()
cmd:option('-loadModel',      'none', 'Provide full path to a trained model')
cmd:option('-optimize',         true, 'Optimize network memory usage.')
cmd:text()
cmd:text(' ---------- Display options --------------------------------------')
cmd:text()
cmd:option('-plotSave',      true, 'Save plot to file (true/false)')
cmd:text(' ---------- Data options ---------------------------------------')
cmd:text()
cmd:text()
cmd:option('-inputRes',          256, 'Input image resolution')
cmd:option('-outputRes',          64, 'Output heatmap resolution')
cmd:text()

opt = cmd:parse(arg or {})
-- add commandline specified options
opt.expDir = paths.concat(opt.expDir, opt.dataset)
opt.save = paths.concat(opt.expDir, opt.expID)

if opt.loadModel == 'none' or opt.loadModel == '' then 
    opt.loadModel = 'final_model.t7'
else
    local str = string.split(opt.loadModel, '/')
    opt.save = paths.concat(opt.expDir, str[#str])
end

if opt.predictions==0 then
    opt.setname = 'val'
else
    opt.setname = 'test'
end
  

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)


-------------------------------------------------------------------------------
-- Load Dataset
-------------------------------------------------------------------------------

paths.dofile('dataset.lua')
g_dataset = loadDataset() -- load dataset train+val+test sets

if not outputDim then
    local nJoints = g_dataset.val.keypoint[1]:size(1)/3
    outputDim = {nJoints, opt.outputRes, opt.outputRes}
end


--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------

if opt.GPU >= 1 then 
    opt.dataType = 'torch.CudaTensor'  -- Use GPU
else
    opt.dataType = 'torch.FloatTensor' -- Use CPU
end

model = torch.load(paths.concat(opt.expDir, opt.expID, opt.loadModel))
model = utils.loadDataParallel(model, 1) -- load model into 'N' GPUs


-- convert modules to a specified tensor type
function cast(x) return x:type(opt.dataType) end  

cast(model) -- convert network's modules data type

if opt.optimize then
   -- for memory optimizations and graph generation
   print('Optimize (reduce) network\'s memory usage...')
   local optnet = require 'optnet'
  
   local sample_input = torch.randn(1,3,opt.inputRes,opt.inputRes):float()
   if opt.GPU>=1 then sample_input=sample_input:cuda() end
   optnet.optimizeMemory(model, sample_input, {inplace = true, reuseBuffers = true, mode = 'inference'})
   print('Done.')
end