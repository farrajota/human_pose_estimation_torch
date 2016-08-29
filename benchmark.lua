--[[
    Test and evaluate trained networks on the MPII or FLIC dataset wrt their PCKh(0.5) performance. (default threshold = 0.5)
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

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
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
cmd:option('-expID',       'default', 'Experiment ID')
cmd:option('-dataset',        'flic', 'Dataset choice: mpii | flic')
cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
cmd:option('-task',       'eval', 'Task to perform: eval, predict-test, predict-valid or demo')
cmd:option('-manualSeed',      2, 'Manually set RNG seed')
cmd:option('-GPU',            -1, 'Default preferred GPU (-1 = use CPU)')
cmd:option('-nGPU',            1, 'Number of GPUs to use by default')
cmd:option('-threshold',     0.5, 'PCKh threshold (default 0.5)')
cmd:text()
cmd:text(' ---------- Model options --------------------------------------')
cmd:text()
cmd:option('-loadModel',      'none', 'Provide full path to a trained model')
cmd:text()
cmd:text(' ---------- Display options --------------------------------------')
cmd:text()
cmd:option('-plotSave',      true, 'Save plot to file (true/false)')
cmd:text()

local opt = cmd:parse(arg or {})
-- add commandline specified options
opt.expDir = paths.concat(opt.expDir, opt.dataset)
opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
opt.save = paths.concat(projectDir, 'results', opt.expID)
if opt.loadModel == 'none' then opt.loadModel = 'final_model.t7' end

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

--------------------------------------------------------------------------------
-- Load/setup data
--------------------------------------------------------------------------------

function loadAnnotations(dataset, set)
    -- Load up a set of annotations for either: 'train', 'valid', or 'test'
    -- There is no part information in 'test'
    -- Flic valid and test set are the same.

    if dataset == 'flic' then
      if set == 'test' then set = 'valid' end
    end
    
    local a = hdf5.open(paths.concat(projectDir,dataset,'annot/' .. set .. '.h5'))
    annot = {}

    -- Read in annotation information from hdf5 file
    local tags = {'part','center','scale','normalize','torsoangle','visible'}
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    annot.nsamples = annot.part:size()[1]
    a:close()

    -- Load in image file names
    -- (workaround for not being able to read the strings in the hdf5 file)
    annot.images = {}
    local toIdxs = {}
    local namesFile = io.open(paths.concat(projectDir,dataset,'annot/' .. set .. '_images.txt'))
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

-- Setup data
if opt.task == 'predict-test' or opt.task = 'demo' then
  -- Test set annotations do not have ground truth part locations, but provide
  -- information about the location and scale of people in each image.
  a = loadAnnotations('test')
elseif opt.task == 'predict-valid' or opt.task == 'eval' then
  -- Validation set annotations on the other hand, provide part locations,
  -- visibility information, normalization factors for final evaluation, etc.
  a = loadAnnotations('valid')
else
  error('Undefined task: ' .. opt.task)
end

if opt.task == 'demo' then
    -- If all the MPII images are available, use the following line to see a random sampling of images
    idxs = torch.randperm(a.nsamples):sub(1,10)
else
    idxs = torch.range(1,a.nsamples)
end

if opt.task == 'eval' then
    nsamples = 0
else
    nsamples = idxs:nElement() 
    -- Displays a convenient progress bar
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,16,2)
end

-- Load model
if opt.GPU >= 1 then 
  -- Use GPU
  require 'cutorch'
  require 'cunn'
  pcall(require, 'cudnn')
  opt.dataType = 'torch.CudaTensor'
else
  -- Use CPU
  opt.dataType = 'torch.FloatTensor'
end

local model = torch.load(opt.loadModel)

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end  

cast(model) -- convert network's modules data type

--------------------------------------------------------------------------------
-- Process dataset
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image
    local im = image.load('images/' .. a['images'][idxs[i]])
    local center = a['center'][idxs[i]]
    local scale = a['scale'][idxs[i]]
    local inp = crop(im, center, scale, 0, 256)

    -- Get network output
    local out = model:forward(cast(inp:view(1,3,256,256)))
    local hm = out[2][1]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img = getPreds(hm, center, scale)
    preds[i]:copy(preds_img)

    xlua.progress(i,nsamples)

    -- Display the result
    if opt.task == 'demo' then
        preds_hm:mul(4) -- Change to input scale
        local dispImg = drawOutput(inp, hm, preds_hm[1])
        w = image.display{image=dispImg,win=w}
        sys.sleep(3)
        if opt.plotSave then image.save(paths.concat(opt.save, 'plot' .. a['images'][idxs[i]]), dispImg) end
    end

    collectgarbage()
end

-- Save predictions
if opt.task == 'predict-valid' then
    local predFile = hdf5.open(paths.concat(opt.save, 'preds-valid-example.h5', 'w'))
    predFile:write('preds', preds)
    predFile:close()
elseif opt.task == 'predict-test' then
    local predFile = hdf5.open(paths.concat(opt.save, 'preds-test.h5', 'w'))
    predFile:write('preds', preds)
    predFile:close()
elseif opt.task == 'demo' then
    w.window:close()
end

--------------------------------------------------------------------------------
-- Evaluation
--------------------------------------------------------------------------------

if opt.task == 'eval' then
    -- Calculate distances given each set of predictions
    local labels = {'valid-example','valid-ours'}
    local dists = {}
    for i = 1,#labels do
        local predFile = hdf5.open(paths.concat(opt.save, 'preds-' .. labels[i] .. '.h5','r')
        local preds = predFile:read('preds'):all()
        table.insert(dists,calcDists(preds, a.part, a.normalize))
    end

    require 'gnuplot'
    gnuplot.raw('set bmargin 1')
    gnuplot.raw('set lmargin 3.2')
    gnuplot.raw('set rmargin 2')    
    gnuplot.raw('set multiplot layout 2,3 title "MPII Validation Set Performance (PCKh)"')
    gnuplot.raw('set xtics font ",6"')
    gnuplot.raw('set ytics font ",6"')
    displayPCK(dists, {9,10}, labels, 'Head')
    displayPCK(dists, {2,5}, labels, 'Knee')
    displayPCK(dists, {1,6}, labels, 'Ankle')
    gnuplot.raw('set tmargin 2.5')
    gnuplot.raw('set bmargin 1.5')
    displayPCK(dists, {13,14}, labels, 'Shoulder')
    displayPCK(dists, {12,15}, labels, 'Elbow')
    displayPCK(dists, {11,16}, labels, 'Wrist', true)
    gnuplot.raw('unset multiplot')
    
    gnuplot.pngfigure(paths.concat(opt.save, opt.dataset .. '_Validation_Set_Performance_PCKh.png')) 
    gnuplot.plotflush()
end

print('Benchmark script complete.')