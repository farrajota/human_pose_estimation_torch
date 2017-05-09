--[[
    Human pose prediction demo. (available datasets: flic, lsp, mpii)
]]

--------------------------------------------------------------------------------
-- Initializations
--------------------------------------------------------------------------------

require 'paths'
require 'torch'
require 'image'
require 'xlua'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'

paths.dofile('projectdir.lua') -- Project directory
paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/draw.lua')
paths.dofile('util/utils.lua') -- for loading the networks

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
paths.dofile('projectdir.lua')
paths.dofile('modules/NoBackprop.lua')


if not pcall(require, 'qt') then
    display = require 'display'
end
  

--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MPII/FLIC Benchmark (PCKh evaluation) options: ')
cmd:text()
cmd:text(' ---------- General options --------------------------------------')
cmd:text()
cmd:option('-expID',       'hg-generic8-teste', 'Experiment ID')
cmd:option('-dataset',        'lsp', 'Datasets: mpii | flic | lsp | mscoco')
cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
cmd:option('-manualSeed',      2, 'Manually set RNG seed')
cmd:option('-GPU',             1, 'Default preferred GPU (-1 = use CPU)')
cmd:option('-nGPU',            1, 'Number of GPUs to use by default')
cmd:text()
cmd:text(' ---------- Model options --------------------------------------')
cmd:text()
cmd:option('-loadModel',      'none', 'Provide the name of a trained model')
cmd:option('-optimize',        'true', 'Optimize network memory usage.')
cmd:text()
cmd:text(' ---------- Display options --------------------------------------')
cmd:text()
cmd:option('-plotSave',      'false', 'Save plot to file (true/false)')
cmd:text(' ---------- Data options ---------------------------------------')
cmd:text()
cmd:option('-nsamples',           10, 'Number of samples to plot')
cmd:option('-inputRes',          256, 'Input image resolution')
cmd:option('-outputRes',          64, 'Output heatmap resolution')
cmd:text()

opt = cmd:parse(arg or {})
-- add commandline specified options
opt.plotSave = (opt.plotSave == 'true')
opt.optimize = (opt.optimize == 'true')
opt.expDir = paths.concat(opt.expDir, opt.dataset)
opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
opt.save = paths.concat(opt.expDir, opt.expID)

if opt.loadModel == 'none' then 
    opt.loadModel = 'final_model.t7'
end

-- Random number seed
if opt.manualSeed ~= -1 then 
    torch.manualSeed(opt.manualSeed)
else 
    torch.seed() 
end

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)


--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------

if opt.GPU >= 1 then 
    opt.dataType = 'torch.CudaTensor'  -- Use GPU
else
    opt.dataType = 'torch.FloatTensor' -- Use CPU
end

local utils = paths.dofile('util/utils.lua')
if opt.GPU >= 1 then 
    opt.dataType = 'torch.CudaTensor'  -- Use GPU
else
    opt.dataType = 'torch.FloatTensor' -- Use CPU
end

local model = torch.load(paths.concat(opt.expDir, opt.expID, opt.loadModel))
model = utils.loadDataParallel(model:clearState(), 1) -- load model into 'N' GPUs

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end  

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



--------------------------------------------------------------------------------
-- Process examples
--------------------------------------------------------------------------------

print('\n==============================================')
print(('Selected dataset: %s'):format(opt.dataset))
print('==============================================\n')

paths.dofile('data.lua')
paths.dofile('dataset.lua')

dataset = loadDataset() -- load dataset (train+val+test sets)

local labels = {'val'}

for k, set in pairs(labels) do
    print(('Processing set: *%s*'):format(set))
    
    -- alloc memory in the gpu for faster data transfer
    local input = torch.Tensor(1,3,opt.inputRes, opt.inputRes); input=cast(input)
    
    local mode = set
    local data = dataset[mode]
    
    for i = 1,opt.nsamples do
        -- get random idx
        local idx = torch.random(1, data.object:size(1))
        
        -- Set up input image
        local im, keypoints, center, scale, _ = loadDataBenchmark(idx, data, mode)
        
        -- Get network output
        input[1]:copy(im) -- copy data from CPU to GPU
        local out = model:forward(input)
        local hm = out[#out][1]:float()
        hm[hm:lt(0)] = 0
          
        -- Get predictions (hm and img refer to the coordinate space)
        local preds_hm, preds_img = getPredsBenchmark(hm, center, scale)
        
        -- Display the result
        preds_hm:mul(4) -- Change to input scale
       -- local dispImg = drawOutput(im, hm, preds_hm[1])
        local heatmaps = drawImgHeatmapParts(im, hm)
        
        -- get crop window
        local crop_coords = im[1]:gt(0):nonzero()
        local center_x_slice = im[{{1},{opt.inputRes/2},{}}]:squeeze():gt(0):nonzero()
        local x1, x2 = center_x_slice:min(), center_x_slice:max()
        local center_y_slice = im[{{1},{},{opt.inputRes/2}}]:squeeze():gt(0):nonzero()
        local y1, y2 = center_y_slice:min(), center_y_slice:max()
        
        local heatmaps_disp = {image.crop(im, x1, y1, x2, y2)}
        for i=1, #heatmaps do
            table.insert(heatmaps_disp, image.crop(image.scale(heatmaps[i], opt.inputRes),x1,y1,x2,y2))
        end
        
        -- display.image(dispImg,{title='image_'..i})
        if pcall(require, 'qt') then
            image.display({image = heatmaps_disp, title='heatmaps_image_'..i})
        else
            display.image(heatmaps_disp,{title='heatmaps_image_'..i})
        end
        
        --sys.sleep(3)
        if opt.plotSave then
            if not paths.dirp(paths.concat(opt.save, 'plot')) then 
                print('Saving plots to: ' .. paths.concat(opt.save, 'plot'))
                os.execute('mkdir -p ' .. paths.concat(opt.save, 'plot'))
            end
            
            image.save(paths.concat(opt.save, 'plot','sample_' .. idx..'.png'), heatmaps_disp[1]) 
            for j=2, #heatmaps_disp do
                image.save(paths.concat(opt.save, 'plot', 'sample_'.. idx..'_heatmap_'..(j-1)..'.png'), heatmaps_disp[j]) 
            end
        end
        
        xlua.progress(i,opt.nsamples)
        
        collectgarbage()
    end
end

print('Demo script complete.')