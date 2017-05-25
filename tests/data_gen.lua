--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
disp = require 'display'

torch.manualSeed(4)

paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'lsp'
opt.rotRate=0

paths.dofile('../dataset.lua')
paths.dofile('../data.lua')

dataset = loadDataset() -- load dataset (train+val+test sets)

--a,b = loadDataBenchmark(1,dataset.val, 'val')
mode = 'train'
--a,b = loadData(1,dataset[mode], mode)
N = 9
local img = torch.FloatTensor(N,3,256,256);
local hm = {}
for i=1, N do
  print(i)
  img[i], hm[i] = loadData(3500,dataset[mode], mode)
end

disp.image(img)

function drawHeatmapParts(input, hms)

    local function colorHM(x)
        -- Converts a one-channel grayscale image to a color heatmap image
        local function gauss(x,a,b,c)
            return torch.exp(-torch.pow(torch.add(x,-b),2):div(2*c*c)):mul(a)
        end
        local cl = torch.zeros(3,x:size(1),x:size(2))
        cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[2] = gauss(x,1,.5,.3)
        cl[3] = gauss(x,1,.2,.3)
        cl[cl:gt(1)] = 1
        return cl
    end

    local colorHms = {}
    local inp64 = input:clone():mul(.3)
    for i = 1,11 do 
        colorHms[i] = colorHM(image.scale(hms[i],256))
        colorHms[i]:mul(.7):add(inp64)
    end
    return colorHms
end

disp.image(drawHeatmapParts(img[1]:double(), hm[1]:double()))

aqui=1

--print('Input size')
--print(#a)
--print('heatmap size')
--print(#b)

