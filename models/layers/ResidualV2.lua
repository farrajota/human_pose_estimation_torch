local conv = nn.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nn.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut,3,3,1,1,1,1))
        :add(batchnorm(numOut))
        :add(relu(true))
        :add(conv(numOut,numOut,3,3,1,1,1,1))
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
function ResidualV2(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

return ResidualV2