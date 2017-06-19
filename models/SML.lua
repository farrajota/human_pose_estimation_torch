--[[
    This network is coined SML (Small-Medium-Large). It is a variation of the stacked hourglass
    with our modified version. It should run faster than the modified hg due to having less convolutions
    and less stacks, concentrating the bulk of the processing needs on 1/2 networks instead of spreading
    it through 8+ stacks of networks.

    This network is designed/composed by three stacked auto-encoders:
    - (S) an auto-encoder with fewer parameters;
      (Note: This gives as an initial/rough estimation of the body parts regions.)

    - (M) has more layers+parameters than the (S) network;
      (Note: This improves the intial estimation and feeds a better map to the next and final layer)

    - (L) has the most layers+parameters+lowest resolution of all networks. Also, it produces the final output.
      (Note: This final layer contains more parameters than the two previous networks combined.
      The resulting output map should be the most accurate map of them all.)


    TODO: complete the code for this network.
]]


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

------------------------------------------------------------------------------------------------------------

local function ae_small(n, f, input)
end

------------------------------------------------------------------------------------------------------------

local function ae_medium(n, f, input)
end

------------------------------------------------------------------------------------------------------------

local function ae_large(n, f, input)
end

------------------------------------------------------------------------------------------------------------

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

------------------------------------------------------------------------------------------------------------

local function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nn.SpatialConvolution(3,64,3,3,1,1,1,1)(inp)           -- 256
    local cnv1 = nn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local pool = nn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 128
    local r1 = Residual(64,128)(cnv1)
    local pool = nn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,256)(r4)
    local up1 = nn.SpatialUpSamplingNearest(2)(low3)                     -- 128
    local inter1 = nn.CAddTable()({inter, hg})
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)                     -- 256
    local inter1 = nn.CAddTable()({inter, hg})

    -- add first AE (Small - 1st rough estimate)
    local hg1 = ae_small(4,256,256)

    -- Linear layers to produce first set of predictions
    local l1 = lin(512,512,hg1)
    local l2 = lin(512,256,l1)

    -- First predicted heatmaps
    local out1 = nn.SpatialConvolution(256,outputDim[1][1],1,1,1,1,0,0)(l2)
    local out1_ = nn.SpatialConvolution(outputDim[1][1],256+128,1,1,1,1,0,0)(out1)

    -- Concatenate with previous linear features
    local cat1 = nn.JoinTable(2)({l2,pool})
    local cat1_ = nn.SpatialConvolution(256+128,256+128,1,1,1,1,0,0)(cat1)
    local int1 = nn.CAddTable()({cat1_,out1_})

    -- add second AE (Medium - better refinement)
    local hg2 = ae_medium(5,256,256+128)
    -- add third AE (Large - best accuracy)
    local hg3 = ae_large(5,256+128,512)

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

    -- Final model
    local model = nn.gModule({inp}, out)

    return model
end

------------------------------------------------------------------------------------------------------------

return createModel