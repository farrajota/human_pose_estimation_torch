paths.dofile('../layers/Residual.lua')

local function hourglass(n, numIn, numOut, inp)
    -- Upper branch
    local up1 = Residual(numIn,256)(inp)
    local up2 = Residual(256,256)(up1)
    local up4 = Residual(256,numOut)(up2)

    -- Lower branch
    local pool = nn.SpatialMaxPooling(2,2,2,2)(inp)
    local low1 = Residual(numIn,256)(pool)
    local low2 = Residual(256,256)(low1)
    local low5 = Residual(256,256)(low2)
    local low6
    if n > 1 then
        low6 = hourglass(n-1,256,numOut,low5)
    else
        low6 = Residual(256,numOut)(low5)
    end
    local low7 = Residual(numOut,numOut)(low6)
    local up5 = nn.SpatialUpSamplingNearest(2)(low7)

    -- Bring two branches together
    return nn.CAddTable()({up4,up5})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

local function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,128)(r4)
    local r6 = Residual(128,256)(r5)

    local output = {}
    local inter = r6

    for i = 1,opt.nStack do
        local iniFeats = 256+(i-1)*128
        local endFeats = 256+i*128
        local hg = hourglass(4,iniFeats,endFeats,inter)

        -- Linear layer to produce first set of predictions
        local l1 = lin(endFeats,endFeats,hg)
        local l2 = lin(endFeats,iniFeats,l1)

        -- First predicted heatmaps
        local out = nn.SpatialConvolution(iniFeats,outputDim[1][1],1,1,1,1,0,0)(l2)
        table.insert(output,out)
        
        if i < opt.nStack then
            local out_ = nn.SpatialConvolution(outputDim[1][1],iniFeats+128,1,1,1,1,0,0)(out)

            -- Concatenate with previous linear features
            local cat = nn.JoinTable(2)({l2,pool})
            local cat_ = nn.SpatialConvolution(iniFeats+128,iniFeats+128,1,1,1,1,0,0)(cat)
            inter = nn.CAddTable()({cat_,out_})
        end
        
    end

    -- Final model
    local model = nn.gModule({inp}, output)

    return model
end

-------------------------

return createModel