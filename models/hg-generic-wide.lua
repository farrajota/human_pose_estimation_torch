paths.dofile('layers/WideResidual.lua')
paths.dofile('layers/Residual.lua')

local function hourglass(n, f, depth, inp)
    -- Upper branch
    local up1 = WideResidual(f,f, depth, 1)(inp)

    -- Lower branch
    local pool = nn.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = WideResidual(f,f, depth, 1)(pool)
    local low2

    if n > 1 then low2 = hourglass(n-1,f,depth,low1)
    else low2 = WideResidual(f,f, depth, 1)(low1) end

    local low3 = WideResidual(f,f, depth, 1)(low2)
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

local function createModel()

    local inp = nn.Identity()()
    
    local depth = 2 --(depth-4)/6
    local widening = 2

    -- Initial processing of the image
    local cnv1_ = nn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local cnv2_ = nn.SpatialConvolution(64,64*widening,1,1,1,1)(cnv1)           -- 128
    local cnv2 = nn.ReLU(true)(nn.SpatialBatchNormalization(64*widening)(cnv2_))
    local r1 = WideResidual(64*widening,128*widening, depth, 2)(cnv2)   -- 64     -- tinha 2 em vez de 1
    local r4 = WideResidual(128*widening,128*widening, depth, 1)(r1)
    local r5 = WideResidual(128*widening,opt.nFeats*widening, depth, 1)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats*widening,depth,inter)

        -- Linear layer to produce first set of predictions
        local ll = lin(opt.nFeats*widening,opt.nFeats*widening,hg)

        -- Predicted heatmaps
        local tmpOut = nn.SpatialConvolution(opt.nFeats*widening,outputDim[1][1],1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        if i < opt.nStack then inter = nn.CAddTable()({inter, hg}) end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model
end

-------------------------

return createModel