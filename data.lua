--[[
    Data sampling functions.
]]


require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/utils.lua')

-------------------------------------------------------------------------------
-- Load in annotations
-------------------------------------------------------------------------------

annotLabels = {'train', 'valid'}
annot,ref = {},{}
for _,l in ipairs(annotLabels) do
    local a, namesFile
    if opt.dataset == 'mpii' and l == 'valid' and opt.finalPredictions == 1 then
        a = hdf5.open(opt.dataDir .. '/annot/test.h5')
        namesFile = io.open(opt.dataDir .. '/annot/test_images.txt')
    else
        a = hdf5.open(opt.dataDir .. '/annot/' .. l .. '.h5')
        namesFile = io.open(opt.dataDir .. '/annot/' .. l .. '_images.txt')
    end
    annot[l] = {}

    -- Read in annotation information
    local tags = {'part', 'center', 'scale', 'normalize', 'torsoangle', 'visible'}
    for _,tag in ipairs(tags) do annot[l][tag] = a:read(tag):all() end
    annot[l]['nsamples'] = annot[l]['part']:size()[1]

    -- Load in image file names (reading strings wasn't working from hdf5)
    annot[l]['images'] = {}
    local toIdxs = {}
    local idx = 1
    for line in namesFile:lines() do
        annot[l]['images'][idx] = line
        if not toIdxs[line] then toIdxs[line] = {} end
        table.insert(toIdxs[line], idx)
        idx = idx + 1
    end
    namesFile:close()

    -- This allows us to reference multiple people who are in the same image
    annot[l]['imageToIdxs'] = toIdxs

    -- Set up reference for training parameters
    ref[l] = {}
    ref[l].nsamples = annot[l]['nsamples']
    ref[l].iters = opt[l .. 'Iters']
    ref[l].batchsize = opt[l .. 'Batch']
end

ref.predict = {}
ref.predict.nsamples = annot.valid.nsamples
ref.predict.iters = annot.valid.nsamples
ref.predict.batchsize = 1

-- Default input is assumed to be an image and output is assumed to be a heatmap
-- This can change if an hdf5 file is used, or if opt.task specifies something different
nParts = annot['train']['part']:size(2)
dataDim = {3, opt.inputRes, opt.inputRes}
labelDim = {nParts, opt.outputRes, opt.outputRes}

-- Load up task specific variables/functions
-- (this allows a decent amount of flexibility in network input/output and training)
paths.dofile('util/' .. opt.task .. '.lua')

function applyFn(fn, t, t2)
    -- Helper function for applying an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

--
-------------------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------------------

-- Manage HDF5 files
useHDF5 = {}
files = {}


for _,l in ipairs({'train','valid'}) do
    useHDF5[l] = false

    if #opt[l .. 'File'] > 0 then
        useHDF5[l] = true
        files[l] = hdf5.open(opt.dataDir .. '/' .. opt[l .. 'File'] .. '.h5', 'r')
        local dataSize = torch.Tensor(files[l]:read('data'):dataspaceSize())
        ref[l]['nsamples'] = dataSize[1]

        -- Use the hdf5 file to determine data/label sizes
        -- These could be anything, not just images/heatmaps
        dataDim = torch.totable(dataSize:sub(2,-1))
        local labelSize = torch.Tensor(files[l]:read('label'):dataspaceSize())
        if labelSize:numel() > 1 then labelDim = torch.totable(labelSize:sub(2,-1))
        else labelDim = {} end

        print("HDF5 file provided (" .. l .. "): " .. opt[l .. "File"])
        print("Number of samples: " .. ref[l]['nsamples'])
    end
end

-- More legible way to print out tensor dimensions
local function print_dims(prefix,d)

end
local function print_dims_old(prefix,d)
    local s = ""
    if #d == 0 then s = "single value"
    elseif #d == 1 then s = string.format("vector of length: %d", d[1])
    else
        s = string.format("tensor with dimensions: %d", d[1])
        for i = 2,table.getn(d) do s = s .. string.format(" x %d", d[i]) end
    end
    print(prefix .. s)
end

function loadData(set, idx, batchsize)
    -- Load in a mini-batch of data
    local input,label

    -- Read data from a provided hdf5 file
    if useHDF5[set] then
        idx = idx or torch.random(annot[set]['nsamples'] - batchsize)
        local inp_dims = {{idx,idx+batchsize-1}}
        for i = 1,#dataDim do inp_dims[i+1] = {1,dataDim[i]} end
        local label_dims = {{idx,idx+batchsize-1}}
        for i = 1,#labelDim do label_dims[i+1] = {1,labelDim[i]} end

        input = files[set]:read('data'):partial(unpack(inp_dims))
        label = files[set]:read('label'):partial(unpack(label_dims))

        if opt.inputRes ~= dataDim[2] or opt.outputRes ~= labelDim[2] then
            -- Data is a fixed size coming from the hdf5 file, so this allows us to resize it
            input = image.scale(input:view(batchsize*dataDim[1],dataDim[2],dataDim[3]),opt.inputRes)
            input = input:view(batchsize,dataDim[1],opt.inputRes,opt.inputRes)
            label = image.scale(label:view(batchsize*labelDim[1],labelDim[2],labelDim[3]),opt.outputRes)
            label = label:view(batchsize,labelDim[1],opt.outputRes,opt.outputRes)
        end

    -- Or generate a new sample
    else
        input = torch.Tensor(batchsize, unpack(dataDim))
        label = torch.Tensor(batchsize, unpack(labelDim))
        for i = 1, batchsize do
            idx_ = idx or torch.random(annot[set]['nsamples'])
            idx_ = (idx_ + i - 2) % annot[set]['nsamples'] + 1
            input[i],label[i] = generateSample(set, idx_)
        end
    end

    if input:max() > 2 then
       input:div(255)
    end

    -- Augment data (during training only)
    if not isTesting then
        local s = torch.randn(batchsize):mul(opt.scaleFactor):add(1):clamp(1-opt.scaleFactor,1+opt.scaleFactor)
        local r = torch.randn(batchsize):mul(opt.rotFactor):clamp(-2*opt.rotFactor,2*opt.rotFactor)

        for i = 1, batchsize do
            -- Color
            input[{i, 1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
            input[{i, 2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
            input[{i, 3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)

            -- Scale/rotation
            if torch.uniform() <= .6 then r[i] = 0 end
            local inp,out = opt.inputRes, opt.outputRes
            input[i] = crop(input[i], {(inp+1)/2,(inp+1)/2}, inp*s[i]/200, r[i], inp)
            label[i] = crop(label[i], {(out+1)/2,(out+1)/2}, out*s[i]/200, r[i], out)
        end

        -- Flip
        local flip_ = customFlip or flip
        local shuffleLR_ = customShuffleLR or shuffleLR
        if torch.uniform() <= .5 then
            input = flip_(input)
            label = flip_(shuffleLR_(label))
        end
    end

    -- Do task-specific preprocessing
    if preprocess then input,label = preprocess(input,label,batchsize,set,idx) end

    return input, label
end

-- Check data preprocessing if there is any
if not alreadyChecked then

  if preprocess then
      print_dims("Original input is a ", dataDim)
      print_dims("Original output is a ", labelDim)
      --print("After preprocessing ---")
      local temp_input,temp_label = loadData('train',1,1)
      -- Input
      if type(temp_input) == "table" then
          inputDim = {}
          --print("Input is a table of %d values" % table.getn(temp_input))
          for i = 1,#temp_input do
              inputDim[i] = torch.totable(temp_input[i][1]:size())
              print_dims("Input %d is a "%i, inputDim[i])
          end
      else
          inputDim = torch.totable(temp_input[1]:size())
          print_dims("Input is a ", inputDim)
      end

      -- Output
      if type(temp_label) == "table" then
          outputDim = {}
          --print("Output is a table of %d values" % #temp_label)
          --print(("Output is a table of %d values"):format(#temp_label))
          for i = 1,#temp_label do
              outputDim[i] = torch.totable(temp_label[i][1]:size())
              --print_dims("Output %d is a "%i, outputDim[i])
              print_dims(("Output %d is a "):format(i), outputDim[i])
          end
      else
          outputDim = torch.totable(temp_label[1]:size())
          print_dims("Output is a ", outputDim)
      end
  else
      inputDim = dataDim
      outputDim = labelDim
      print_dims("Input is a ", inputDim)
      print_dims("Output is a ", outputDim)
  end

end
