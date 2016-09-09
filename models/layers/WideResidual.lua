--[[
    Wide residual networks blocks.
    Source: https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
]]

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local dropout = 0 -- dropout disabled

local function Dropout()
      return nn.Dropout(opt and opt.dropout or 0,nil,true)
end

local function WideBasic(nInputPlane, nOutputPlane, stride)
    local conv_params = {
        {3,3,stride,stride,1,1},
        {3,3,1,1,1,1},
    }
    local nBottleneckPlane = nOutputPlane

    local block = nn.Sequential()
    local convs = nn.Sequential()     

    for i,v in ipairs(conv_params) do
        if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
        else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
        end
    end
    
    local shortcut = nInputPlane == nOutputPlane and nn.Identity() or Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
     
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
end

 -- Stacking Residual Units on the same stage
function WideResidual(numIn, numOut, count, stride)
    local s = nn.Sequential()
    
    local block = WideBasic
    
    s:add(block(numIn, numOut, stride))
    for i=2,count do
        s:add(block(numOut, numOut, 1))
    end
    return s
end
