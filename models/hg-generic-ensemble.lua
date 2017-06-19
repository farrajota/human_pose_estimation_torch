paths.dofile('layers/ResidualBest.lua')

local function lin_old(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local bn_relu = nn.RReLU()(nn.SpatialBatchNormalization(numIn)(inp))
    local spatial_dropout = nn.SpatialDropout(opt.spatialdropout)(bn_relu)
    return nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(spatial_dropout)
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local bn_relu = nn.ReLU(true)(nn.SpatialBatchNormalization(numIn)(inp))
    return nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(bn_relu)
end

local function createModel()

    if not nn.NoBackprop then
        paths.dofile('modules/NoBackprop.lua')
    end

    print('Load model: ' .. paths.concat(opt.ensemble, 'final_model.t7'))
    local trained_model = torch.load(paths.concat(opt.ensemble, 'final_model.t7'))
    trained_model:evaluate()

    -- craft network
    local inp = nn.Identity()()
    local hg_net = nn.NoBackprop(trained_model)(inp)  -- disable backprop
    local concat_outputs = nn.JoinTable(2)(hg_net)
    local ll1 = lin(outputDim[1][1]*opt.nStack, 512, concat_outputs)
    local out = lin(512, outputDim[1][1], ll1)

    opt.nOutputs = 1

    -- Final model
    local model = nn.gModule({inp}, {out})

    return model
end

-------------------------

return createModel