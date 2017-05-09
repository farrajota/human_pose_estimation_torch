paths.dofile('layers/ResidualBest.lua')

local feat_inc = 96 --128

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = Residual(f,f)(inp)

    -- Lower branch
    local pool = nn.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(f,f+feat_inc)(pool)
    local low2

    if n > 1 then low2 = hourglass(n-1,f+feat_inc,low1)
    else low2 = Residual(f+feat_inc,f+feat_inc)(low1) end

    local low3 = Residual(f+feat_inc,f)(low2)
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nn.RReLU()(nn.SpatialBatchNormalization(numOut)(l))
end

local function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,256)(r4)
    local r6 = Residual(256,opt.nFeats)(r5)

    local out = {}
    local inter = r6

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Linear layer to produce first set of predictions
        local spDropout_ll = nn.SpatialDropout(opt.spatialdropout)(hg)
        local ll = lin(opt.nFeats,opt.nFeats,spDropout_ll)

        -- Predicted heatmaps
        local spDropout_Out = nn.SpatialDropout(opt.spatialdropout)(ll)
        local tmpOut = nn.SpatialConvolution(opt.nFeats,outputDim[1][1],1,1,1,1,0,0)(spDropout_Out)
        
        if i > 1 then
            local concat = nn.JoinTable(2)({tmpOut,out[#out]})
            local tmpOut2 = nn.SpatialConvolution(outputDim[1][1]*2,outputDim[1][1],1,1,1,1,0,0)(concat)
            table.insert(out,tmpOut2)
        else
            table.insert(out,tmpOut)
        end

        if i < opt.nStack then inter = nn.CAddTable()({inter, hg}) end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model
end

-------------------------

return createModel