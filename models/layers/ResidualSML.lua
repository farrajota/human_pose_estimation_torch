local function select_nonlinearity()
    local str = string.lower(opt.nonlinearity)
    if str == 'relu' then
        return nn.ReLU(true)
    elseif str == 'elu' then
        return nn.ELU(nil, true)
    elseif str == 'prelu' then
        return nn.PReLU()
    elseif str == 'rrelu' then
        return nn.RReLU(nil, nil, true)
    elseif str == 'leakyrelu' then
        return nn.LeakyReLU(nil, true)
    else
        error('Invalid non-linearity: ' .. opt.nonlinearity)
    end
end

local function select_dropout()
    if opt.spatialdropout > 0 then
        return nn.SpatialDropout(opt.spatialdropout)
    elseif opt.dropout > 0 then
        return nn.Dropout(opt.dropout)
    else
        return nn.Identity()
    end
end

local function main_conv(op, numIn,numOut, kW, kH, dW, dH, padW, padH)
    if op == 'D' then
        return nn.SpatialConvolution(numIn, numOut, kW, kH, 2, 2, padW, padH)
    elseif op == 'U' then
        return nn.SpatialFullConvolution(numIn, numOut, kW, kH, 2, 2, padW, padH, 1, 1)
    else
        return nn.SpatialConvolution(numIn, numOut, kW, kH, dW, dH, padW, padH)
    end
end


local conv = nn.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = select_nonlinearity
local dropout = select_dropout


-- Main convolutional block
local function convBlock(numIn,numOut, op)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu())
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu())
        :add(dropout())
        :add(main_conv(op,numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu())
        :add(conv(numOut/2,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn, numOut, op)
    if numIn == numOut then
        if op == 'D' or op == 'U' then
            return main_conv(op, numIn,numOut,1,1,1,1,0,0)
        else
            return nn.Identity()
        end
    else
        return nn.Sequential()
            :add(main_conv(op, numIn,numOut,1,1,1,1,0,0))
    end
end

-- Residual block
function Residual(numIn, numOut, operation)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut, operation))
            :add(skipLayer(numIn,numOut, operation)))
        :add(nn.CAddTable(true))
end

return Residual, relu