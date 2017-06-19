paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = Residual(f,f)(inp)

    -- Lower branch
    local pool = nn.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(f,f)(pool)
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else low2 = Residual(f,f)(low1) end

    local low3 = Residual(f,f)(low2)
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

local function lin2(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local bn_relu = nn.ReLU(true)(nn.SpatialBatchNormalization(numIn)(inp))
    return nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(bn_relu)
end

local function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Linear layer to produce first set of predictions
        local ll = lin(opt.nFeats,opt.nFeats,hg)

        -- Predicted heatmaps
        local tmpOut = nn.SpatialConvolution(opt.nFeats,outputDim[1][1],1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        if i < opt.nStack then inter = nn.CAddTable()({inter, hg}) end
    end

    local concat_outputs = nn.JoinTable(2)(out)
    local ll1 = lin2(outputDim[1][1]*opt.nStack , 512, concat_outputs)
    local ll2 = lin2(512, outputDim[1][1], ll1)

    table.insert(out, ll2)
    opt.nOutputs = opt.nStack+1

    -- Final model
    local model = nn.gModule({inp}, out)

    return model
end

-------------------------

return createModel