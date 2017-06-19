paths.dofile('layers/Residual.lua')

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local dropout = nn.SpatialDropout(0.2)(inp)
    local bn_relu = nn.ReLU(true)(nn.SpatialBatchNormalization(numIn)(dropout))
    return nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(bn_relu)
end

local function createModel()

    if not nn.NoBackprop then
        paths.dofile('modules/NoBackprop.lua')
    end

    local trained_model = torch.load('/home/mf/Toolkits/Codigo/git/pose-torchnet/exp/flic/hg-generic8/final_model.t7')
    trained_model:evaluate()

    -- craft network
    local inp = nn.Identity()()
    local hg_net = nn.NoBackprop(trained_model)(inp)  -- disable backprop
    local concat_outputs = nn.JoinTable(2)(hg_net)
    local ll1 = lin(outputDim[1][1]*8, 512, concat_outputs)
    local out = lin(512, outputDim[1][1], ll1)

    opt.nOutputs = 1

    -- Final model
    local model = nn.gModule({inp}, {out})

    return model
end

-------------------------

return createModel