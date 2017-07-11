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

      Version 2
]]


local Residual, relu = paths.dofile('../layers/ResidualSML.lua')

------------------------------------------------------------------------------------------------------------

local function beautify_modules(modules)

    -- auto-encoder small
    if modules.ae_small then
        modules.ae_small:annotate{
            name = 'Auto-Encoder (Small)',
            description = 'Auto-Encoder network with I/O resolution 64x64',
            graphAttributes = {
                color = 'red',
                style = 'filled',
                fillcolor = 'yellow'}
        }
    end

    -- auto-encoder medium
    if modules.ae_medium then
        modules.ae_medium:annotate{
            name = 'Auto-Encoder (Medium)',
            description = 'Auto-Encoder network with I/O resolution 128x128',
            graphAttributes = {
                color = 'red',
                style = 'filled',
                fillcolor = 'yellow'}
        }
    end

    -- auto-encoder large
    if modules.ae_large then
        modules.ae_large:annotate{
            name = 'Auto-Encoder (Large)',
            description = 'Auto-Encoder network with I/O resolution 256x256',
            graphAttributes = {
                color = 'red',
                style = 'filled',
                fillcolor = 'yellow'}
        }
    end

    -- linear layers
    if modules.lin then
        for k, lin_layer in pairs(modules.lin) do
            lin_layer:annotate{
                name = 'Linear layer',
                description = 'Linear layer with 1x1 convolutions',
                graphAttributes = {color = 'blue', fontcolor = 'green'}
            }
        end
    end

    -- linear layers
    if modules.out then
        for k, output_layer in pairs(modules.out) do
            output_layer:annotate{
                name = 'Output layer',
                description = 'Output heatmaps',
                graphAttributes = {
                    color = 'black',
                    style = 'filled',
                    fillcolor = 'blue'}
            }
        end
    end
end

------------------------------------------------------------------------------------------------------------

local function AE_small(n, f)
    ----
    local feat_inc = 0--16
    local function hourglass(n, f, inp)
        -- Upper branch
        local up1 = Residual(f,f)(inp)

        -- Lower branch
        local low1 = Residual(f,f+feat_inc, 'D')(inp)
        local low2
        if n > 1 then
            low2 = hourglass(n-1,f+feat_inc,low1)
        else
            low2 = Residual(f+feat_inc,f+feat_inc)(low1)
        end
        local up2 = Residual(f+feat_inc,f, 'U')(low2)

        -- Bring two branches together
        return nn.CAddTable()({up1,up2})
    end
    ----

    local input = nn.Identity()()
    local output = hourglass(n, f, input)
    local network = nn.gModule({input}, {output})
    network.name = 'AE_small'
    return network
end

------------------------------------------------------------------------------------------------------------

local function AE_medium(n, f)
    ----
    local feat_inc = 0 --32
    local function hourglass(n, f, inp)
        -- Upper branch
        local up1 = Residual(f,f+feat_inc)(inp)
        local up2 = Residual(f+feat_inc,f)(up1)

        -- Lower branch
        local low1 = Residual(f,f+feat_inc, 'D')(inp)
        local low2 = Residual(f+feat_inc,f+feat_inc)(low1)
        local low3
        if n > 1 then
            low3 = hourglass(n-1,f+feat_inc, low2)
        else
            low3 = Residual(f+feat_inc, f+feat_inc)(low2)
        end
        local up3 = Residual(f+feat_inc, f, 'U')(low3)

        -- Bring two branches together
        return nn.CAddTable()({up2,up3})
    end
    ----

    local input = nn.Identity()()
    local output = hourglass(n, f, input)
    local network = nn.gModule({input}, {output})
    network.name = 'AE_medium'
    return network
end

------------------------------------------------------------------------------------------------------------

local function AE_large(n, f)
    ----
    local feat_inc = 0 --96
    local function hourglass(n, f, inp)
        -- Upper branch
        local up1 = Residual(f,f+feat_inc)(inp)
        local up2 = Residual(f+feat_inc,f+feat_inc)(up1)
        local up4 = Residual(f+feat_inc,f)(up2)

        -- Lower branch
        local low1 = Residual(f,f+feat_inc, 'D')(inp)
        local low2 = Residual(f+feat_inc,f+feat_inc)(low1)
        local low5 = Residual(f+feat_inc,f+feat_inc)(low2)
        local low6
        if n > 1 then
            low6 = hourglass(n-1,f+feat_inc,low5)
        else
            low6 = Residual(f+feat_inc,f+feat_inc)(low5)
        end
        local up5 = Residual(f+feat_inc,f, 'U')(low6)

        -- Bring two branches together
        return nn.CAddTable()({up4,up5})
    end
    ----

    local input = nn.Identity()()
    local output = hourglass(n, f, input)
    local network = nn.gModule({input}, {output})
    network.name = 'AE_large'
    return network
end

------------------------------------------------------------------------------------------------------------

local function lin(numIn,numOut)
    -- Apply 1x1 convolution, stride 1, no padding
    return nn.Sequential()
        :add(nn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0))
        :add(nn.SpatialBatchNormalization(numOut))
        :add(relu())
end

------------------------------------------------------------------------------------------------------------

local function createModel()
    local input = nn.Identity()()

    -- Initial processing of the image
    local cnv1 = nn.Sequential()
        :add(nn.SpatialConvolution(3,64,7,7,2,2,3,3))     -- 128
        :add(nn.SpatialBatchNormalization(64))
        :add(relu())(input)
    local r1 = Residual(64,128, 'D')(cnv1)                -- 64
    local r2 = Residual(128,128)(r1)
    local r3 = Residual(128,128)(r2)
    local r4 = Residual(128,256)(r3)


    --------------------------------------------------------------------------------
    -- Small Auto-Encoder Network
    --------------------------------------------------------------------------------

    local r4_1 = Residual(256,128)(r4)
    local autoencoder_s = AE_small(4, 128)  -- small AE
    local autoencoder_s_out = autoencoder_s(r4_1)

    -- Linear layers to produce first set of predictions
    local l1_s = lin(128,256)(autoencoder_s_out)
    local l2_s = lin(256,256)(l1_s)

    -- First predicted heatmaps (small AE)
    local out_s = nn.SpatialConvolution(256,outputDim[1][1],1,1,1,1,0,0)(l2_s)
    local out_s_ = nn.SpatialConvolution(outputDim[1][1],256,1,1,1,1,0,0)(out_s)

    -- Concatenate with previous linear features
    local cat1 = nn.JoinTable(2)({l2_s,r4})
    local cat1_ = nn.SpatialConvolution(256+256,256,1,1,1,1,0,0)(cat1)
    local int1 = nn.CAddTable()({cat1_,out_s_})

    -- increase resolution for the second network
    local r5 = Residual(256,256, 'U')(int1)


    --------------------------------------------------------------------------------
    -- Medium Auto-Encoder Network
    --------------------------------------------------------------------------------

    local autoencoder_m = AE_medium(5, 256)  -- medium AE
    local autoencoder_m_out = autoencoder_m(r5)

    -- Linear layers to produce the second set of predictions
    local l1_m = lin(256,512)(autoencoder_m_out)
    local l2_m = lin(512,512)(l1_m)

    -- Second predicted heatmaps (small AE)
    local out_m = nn.SpatialConvolution(512,outputDim[1][1],1,1,1,1,0,0)(l2_m)

    beautify_modules({
        ae_small = autoencoder_s_out,
        ae_medium = autoencoder_m_out,
        lin = {l1_s, l2_s, l1_m, l2_m},
        out = {out_s, out_m}
    })

    -- Final model
    local out_s_up = nn.SpatialUpSamplingNearest(4)(out_s)
    local out_m_up = nn.SpatialUpSamplingNearest(2)(out_m)
    local model = nn.gModule({input}, {out_s_up, out_m_up})

    return model
end

------------------------------------------------------------------------------------------------------------

return createModel