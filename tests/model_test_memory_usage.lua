require 'paths'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
optnet = require 'optnet'

torch.setdefaulttensortype('torch.FloatTensor')

opt = {}
opt.display = false
opt.nStack = 8
opt.dropout = 0
opt.spatialdropout = 0.2
opt.nonlinearity = 'relu'
opt.nFeats = 256
opt.nStack = 8
opt.inputRes = 256
opt.outputRes = 256
outputDim = {{16}, {16}}


--[[ Load model list ]]--
local models = paths.dofile('../models/init.lua')

local test_models = {
    hg_best= models['hg-generic-best'],
    hg_generic8 = models['hg-generic'],
    hg_generic8_fullconv = models['hg-generic-full-conv'],
    hg_stacked = models['hg-stacked'],
    hg_stacked = models['hg-stacked'],
    hg_stacked_fullconv = models['hg-stacked-full-conv'],
    sml_v2 = paths.dofile('../models/test/SML_v2.lua'),
    sml_v3 = paths.dofile('../models/test/SML_v3.lua'),
    sml_v3_featinc = paths.dofile('../models/test/SML_v3_1.lua'),
    sml_v3_lin512 = paths.dofile('../models/test/SML_v3_2.lua'),
    sml_v3_maxpool = paths.dofile('../models/test/SML_v3_3.lua'),
    sml_v3_maxpool_128_256 = paths.dofile('../models/test/SML_v3_4.lua'),
    sml_v3_64x64 = paths.dofile('../models/test/SML_v3_5.lua'),
    sml_v3_64x64_fullconv = paths.dofile('../models/test/SML_v3_6.lua'),
    sml_v4_fullconv = paths.dofile('../models/test/SML_v4.lua'),
    sml_v4_featinc = paths.dofile('../models/test/SML_v4_1.lua'),
    sml_v4_256x256 = paths.dofile('../models/test/SML_v4_2.lua'),
    sml_v4_256x256_maxpool_up = paths.dofile('../models/test/SML_v4_3.lua'),
    sml_v4_256x256_maxpool = paths.dofile('../models/test/SML_v4_4.lua'),
    sml_v5 = paths.dofile('../models/test/SML_v5.lua'),
}


--[[ Setup input tensor ]]--
local input = torch.Tensor(2, 3, 256, 256):uniform()


-- test all architectures
local i = 1
local total_tests = 0
for _, _ in pairs(test_models) do
    total_tests = total_tests + 1
end

summary_output = {}
for model_name, load_model in pairs(test_models) do
    print('\n*********************')
    print(('Test model (%d/%d): %s'):format(i, total_tests, model_name))
    print('*********************\n')
    i = i+1

    --[[ Construct model ]]--
    local model = load_model()
    model:float()

    --[[ Test model ]]--
    nngraph.setDebug(true)
    model.name = model_name


    if pcall(function() model:updateOutput(input) end) then
        print('Model forward pass successful')

        mem1 = optnet.countUsedMemory(model)
        print('Memory usage: ' .. mem1.total_size/1024/1024)
        summary_output[model_name] = mem1.total_size/1024/1024
    else
        print('The forward passed error. Check the model architecture.')
    end


    --[[ display model ]]
    if opt.display then
        if pcall(require, 'qt') then
            local tmp_fname = paths.concat(paths.home, 'tmp', 'graphs', 'sml_network.svg')
            graph.dot(model.fg, 'Forward Graph', tmp_fname)
            graph.dot(model.fg, 'Forward Graph')
            print('Model plot saved to: ' .. tmp_fname)
        else
            print('*** Please run this script with the qlua interpreter to see the model plot ***')
        end
    end

    model=nil
    collectgarbage()
    collectgarbage()
end


print('\n*************************')
print("Memory usage summary:")
print('*************************\n')
for model_name, mem in pairs(summary_output) do
    print('   > ' .. model_name .. ': \t' .. mem)
end