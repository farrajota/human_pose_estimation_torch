--[[
    Demo for the pose detector. (available datasets: flic)
]]


--------------------------------------------------------------------------------
-- Initializations
--------------------------------------------------------------------------------

require 'paths'
require 'torch'
require 'hdf5'
require 'image'
require 'xlua'
require 'nn'
require 'nnx'
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


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MPII/FLIC Benchmark (PCKh evaluation) options: ')
cmd:text()
cmd:text(' ---------- General options --------------------------------------')
cmd:text()
cmd:option('-expID',       'hg-stacked-adam', 'Experiment ID')
cmd:option('-dataset',        'flic', 'Dataset choice: mpii | flic')
cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
cmd:option('-manualSeed',      2, 'Manually set RNG seed')
cmd:option('-GPU',             1, 'Default preferred GPU (-1 = use CPU)')
cmd:option('-nGPU',            1, 'Number of GPUs to use by default')
cmd:text()
cmd:text(' ---------- Model options --------------------------------------')
cmd:text()
cmd:option('-loadModel',      'none', 'Provide full path to a trained model')
cmd:option('-optimize',         true, 'Optimize network memory usage.')
cmd:text()
cmd:text(' ---------- Display options --------------------------------------')
cmd:text()
cmd:option('-plotSave',      true, 'Save plot to file (true/false)')
cmd:option('-nsamples',      50, 'Number of images to plot')
cmd:text(' ---------- Data options ---------------------------------------')
cmd:text()
cmd:text()
cmd:option('-inputRes',          256, 'Input image resolution')
cmd:option('-outputRes',          64, 'Output heatmap resolution')
cmd:text()

local opt = cmd:parse(arg or {})
-- add commandline specified options
opt.expDir = paths.concat(opt.expDir, opt.dataset)
opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
opt.save = paths.concat(opt.expDir, opt.expID)

if opt.loadModel == 'none' then
    opt.loadModel = 'final_model.t7'
else
    local str = string.split(opt.loadModel, '/')
    opt.save = paths.concat(opt.expDir, str[#str])
end

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)
if opt.plotSave then
    print('Saving plots to: ' .. paths.concat(opt.save, 'plot'))
    os.execute('mkdir -p ' .. paths.concat(opt.save, 'plot'))
end

torch.manualSeed(opt.manualSeed)

--------------------------------------------------------------------------------
-- Load/setup data
--------------------------------------------------------------------------------

local function loadAnnotations(set)
    -- Load up a set of annotations for either: 'train', 'valid', or 'test'
    -- There is no part information in 'test'
    -- Flic valid and test set are the same.

    if opt.dataset == 'flic' then
      if set == 'test' then set = 'valid' end
    end

    local a = hdf5.open(paths.concat(projectDir, 'data', opt.dataset, 'annot/' .. set .. '.h5'))
    local annot = {}

    -- Read in annotation information from hdf5 file
    local tags = {'part','center','scale','normalize','torsoangle','visible'}
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    annot.nsamples = annot.part:size()[1]
    a:close()

    -- Load in image file names
    -- (workaround for not being able to read the strings in the hdf5 file)
    annot.images = {}
    local toIdxs = {}
    local namesFile = io.open(paths.concat(projectDir,'data', opt.dataset,'annot/' .. set .. '_images.txt'))
    local idx = 1
    for line in namesFile:lines() do
        annot.images[idx] = line
        if not toIdxs[line] then toIdxs[line] = {} end
        table.insert(toIdxs[line], idx)
        idx = idx + 1
    end
    namesFile:close()

    -- This allows us to reference all people who are in the same image
    annot.imageToIdxs = toIdxs

    return annot
end


--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------

if opt.GPU >= 1 then
    opt.dataType = 'torch.CudaTensor'  -- Use GPU
else
    opt.dataType = 'torch.FloatTensor' -- Use CPU
end

local utils = paths.dofile('util/utils.lua')
local model = utils.loadDataParallel((paths.concat(opt.expDir, opt.expID, opt.loadModel)), opt.nGPU) -- load model into 'N' GPUs
model:evaluate()

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

local labels = {'test'}

for k, set in pairs(labels) do
    print(('Processing set: *%s*'):format(set))
    -- load annotations
    local a = loadAnnotations(set)
    -- define index range of the number of available samples
    local idxs = torch.randperm(a.nsamples):sub(1,opt.nsamples)
    local nsamples = idxs:nElement()
    -- Displays a convenient progress bar
    xlua.progress(0,nsamples)
    local preds = torch.Tensor(nsamples,16,2)

    -- alloc memory in the gpu for faster data transfer
    local input = torch.Tensor(1,3,opt.inputRes, opt.inputRes); input=cast(input)
    for i = 1,nsamples do
        -- Set up input image
        local im = image.load(paths.concat(projectDir, 'data', opt.dataset, 'images/' .. a['images'][idxs[i]]))
        local center = a['center'][idxs[i]]
        local scale = a['scale'][idxs[i]]
        local inp = crop(im, center, scale, 0, 256)

        -- Get network output
        input[1]:copy(inp) -- copy data from CPU to GPU
        local out = model:forward(input)
        local hm = out[2][1]:float()
        hm[hm:lt(0)] = 0

        -- Get predictions (hm and img refer to the coordinate space)
        local preds_hm, preds_img = getPredsBenchmark(hm, center, scale)

        -- Display the result
        preds_hm:mul(4) -- Change to input scale
        local dispImg = drawOutputFLIC(inp, hm, preds_hm[1])
        local skeliImg = drawSkeletonFLIC(inp, hm, preds_hm[1])
        local partsImg = drawHeatmapPartsFLIC(inp, hm, preds_hm[1])
        if pcall(require, 'qt') then image.display{image=dispImg} end
        --sys.sleep(3)
        if opt.plotSave then
            --image.save(paths.concat(opt.save, 'plot', 'plot_' .. a['images'][idxs[i]]), dispImg)
            image.save(paths.concat(opt.save, 'plot', 'Skeleton_' .. a['images'][idxs[i]]), skeliImg)
        if i > 15 and i < 25 then
            image.save(paths.concat(opt.save, 'plot', 'original_' .. a['images'][idxs[i]]), inp)
            for j=1, #partsImg do
                local tmp_img = image.scale(partsImg[j], 378,378)
                image.save(paths.concat(opt.save, 'plot', 'heatmap_part' .. j .. '_' .. a['images'][idxs[i]]), tmp_img)
            end
            end
        end

        xlua.progress(i,nsamples)

        collectgarbage()
    end
end

print('Demo script complete.')