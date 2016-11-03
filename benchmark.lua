--[[
    Predict body positions and evaluate trained networks on the MPII or FLIC dataset wrt to their performance.
]]


--------------------------------------------------------------------------------
-- Initializations
--------------------------------------------------------------------------------

require 'paths'
require 'torch'
require 'image'
require 'xlua'
require 'nn'
require 'nnx'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
local json = require 'json'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/utils.lua') -- for loading the networks
paths.dofile('projectdir.lua') -- Project directory

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
cmd:option('-expID',       'hg-generic-maxpool', 'Experiment ID')
cmd:option('-dataset',        'flic', 'Dataset choice: mpii | flic')
cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
cmd:option('-reprocess',   false,  'Utilize existing predictions from the model\'s folder.')
cmd:option('-manualSeed',      2, 'Manually set RNG seed')
cmd:option('-GPU',             1, 'Default preferred GPU (-1 = use CPU)')
cmd:option('-nGPU',            1, 'Number of GPUs to use by default')
cmd:option('-threshold',     0.5, 'PCKh threshold (default 0.5)')
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
--local model = torch.load(paths.concat(opt.expDir, opt.expID, opt.loadModel))

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
-- Process dataset
--------------------------------------------------------------------------------

print('\n==============================================')
print(('Selected dataset: %s'):format(opt.dataset))
print('==============================================\n')

local labels = {'valid', 'test'}
if opt.dataset == 'flic' then  labels = {'valid'} end

for k, set in pairs(labels) do
    -- check if files already exist, and if the reprocess flag is false
    if opt.reprocess or not paths.filep(paths.concat(opt.save, 'preds-' .. set.. '.h5')) then
        print(('Processing set: *%s*'):format(set))
        -- load annotations
        local a = loadAnnotations(set)
        -- define index range of the number of available samples
        local idxs = torch.range(1,a.nsamples)
        local nsamples = idxs:nElement()
        -- Displays a convenient progress bar
        xlua.progress(0,nsamples)
        
        -- select number of partys depedning on the dataset
        if opt.dataset == 'mpii' then
            nparts = 16
        elseif opt.dataset == 'flic' then
            nparts = 11
        else
            error('Undefined dataset: ' .. opt.dataset)
        end
        
        local preds = torch.Tensor(nsamples,nparts,2)
        
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
            preds[i]:copy(preds_img)
            
            xlua.progress(i,nsamples)
            
            collectgarbage()
        end
        
        -- store predictions to file
        local predFile = hdf5.open(paths.concat(opt.save, 'preds-' .. set.. '.h5'), 'w')
        predFile:write('preds', preds)
        predFile:close()
        
        -- compute/display pck
        
        print('Done.')
    end
end


--------------------------------------------------------------------------------
-- Evaluation
--------------------------------------------------------------------------------

-- Calculate distances given each set of predictions
local dists = {}
for i=1, #labels do
    local a = loadAnnotations(labels[i])
    local predFile = hdf5.open(paths.concat(opt.save, 'preds-' .. labels[i] .. '.h5'),'r')
    local preds = predFile:read('preds'):all():double()
    table.insert(dists,calcDists(preds, a.part, a.normalize))
end

local res = {}
if opt.dataset == 'mpii' then
require 'gnuplot'
gnuplot.raw('set bmargin 1')
gnuplot.raw('set lmargin 3.2')
gnuplot.raw('set rmargin 2')    
gnuplot.raw('set multiplot layout 2,3 title "MPII Validation Set Performance (PCKh)"')
gnuplot.raw('set xtics font ",6"')
gnuplot.raw('set ytics font ",6"')
print('-----------------------------------')
displayPCK(dists, {9,10}, labels, 'Head')
print('-----------------------------------')
displayPCK(dists, {2,5}, labels, 'Knee')
print('-----------------------------------')
displayPCK(dists, {1,6}, labels, 'Ankle')
print('-----------------------------------')
gnuplot.raw('set tmargin 2.5')
gnuplot.raw('set bmargin 1.5')
displayPCK(dists, {13,14}, labels, 'Shoulder')
print('-----------------------------------')
displayPCK(dists, {12,15}, labels, 'Elbow')
print('-----------------------------------')
displayPCK(dists, {11,16}, labels, 'Wrist', true)
print('-----------------------------------')
gnuplot.raw('unset multiplot')

gnuplot.pngfigure(paths.concat(opt.save, 'Validation_Set_Performance_PCKh.png')) 
gnuplot.plotflush()

else
    require 'gnuplot'
    gnuplot.raw('set bmargin 1')
    gnuplot.raw('set lmargin 3.2')
    gnuplot.raw('set rmargin 2')    
    gnuplot.raw('set multiplot layout 2,3 title "FLIC Validation Set Performance (PCKh)"')
    gnuplot.raw('set xtics font ",6"')
    gnuplot.raw('set ytics font ",6"')
    gnuplot.raw('set tmargin 2.5')
    gnuplot.raw('set bmargin 1.5')
    res[1] = displayPCK(dists, {1,4}, labels, 'Shoulder')
    print('-----------------------------------')
    res[2] = displayPCK(dists, {2,5}, labels, 'Elbow')
    print('-----------------------------------')
    res[3] = displayPCK(dists, {3,6}, labels, 'Wrist', true)
    print('-----------------------------------')
    gnuplot.raw('unset multiplot')

    gnuplot.pngfigure(paths.concat(opt.save, 'Validation_Set_Performance_PCKh.png')) 
    gnuplot.plotflush()

end

json.save(paths.concat(opt.save,'Validation_Set_Performance_results.json'), json.encode(res))

print('Benchmark script complete.')