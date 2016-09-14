local conv = nn.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nn.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(nn.SpatialDropout(0.2))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(nn.SpatialDropout(0.2))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

-- Residual block
local function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

 -- Stacking Residual Units on the same stage
function WideResidual(numIn, numOut, count, stride)
    local s = nn.Sequential()
    
    local block = Residual
    
    s:add(block(numIn, numOut))
    for i=2,count do
        s:add(block(numOut, numOut))
    end
    return s
end