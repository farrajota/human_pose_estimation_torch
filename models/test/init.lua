--[[
    Network architectures used in tests.
]]

local function additional_network_architectures(model_list)

    local model_list = model_list or {}

    -- stacked
    model_list['hg-stacked']           = paths.dofile('hg-stacked.lua')
    model_list['hg-stackedV2']         = paths.dofile('hg-stackedV2.lua')
    model_list['hg-stacked-wide']      = paths.dofile('hg-stacked-wide.lua')
    model_list['hg-stacked-no-int']    = paths.dofile('hg-stacked-no-int.lua')
    model_list['hg-stacked-full-conv'] = paths.dofile('hg-stackedFullConv.lua')
    -- generic
    model_list['hg-generic']   = paths.dofile('hg-generic.lua')
    model_list['hg-genericv2'] = paths.dofile('hg-genericv2.lua')
    model_list['hg-genericv3'] = paths.dofile('hg-genericv3.lua')
    model_list['hg-generic-full-conv'] = paths.dofile('hg-genericFullConv.lua')
    -- RRelu
    model_list['hg-generic-rrelu'] = paths.dofile('hg-generic-rrelu.lua')
    -- modified generic
    model_list['hg-generic-deconv']     = paths.dofile('hg-generic-deconv.lua')
    model_list['hg-generic-maxpool']    = paths.dofile('hg-generic-maxpool.lua')
    model_list['hg-generic-deception']  = paths.dofile('hg-generic-deception.lua')
    model_list['hg-generic-inception']  = paths.dofile('hg-generic-inception.lua')
    model_list['hg-generic-wide']       = paths.dofile('hg-generic-wide.lua')
    model_list['hg-generic-widev2']     = paths.dofile('hg-generic-widev2.lua')
    model_list['hg-generic-widev3']     = paths.dofile('hg-generic-widev3.lua')
    model_list['hg-generic-widefeats']  = paths.dofile('hg-generic-widefeats.lua')
    model_list['hg-generic-highres']    = paths.dofile('hg-generic-highres.lua')
    -- rnn generic
    model_list['hg-generic-rnn']        = paths.dofile('hg-generic-rnn.lua')

    -- Concat sets
    model_list['hg-generic-concatv1']     = paths.dofile('hg-generic-concatv1.lua')
    model_list['hg-generic-concatv2']     = paths.dofile('hg-generic-concatv2.lua')
    model_list['hg-generic-concatN']      = paths.dofile('hg-generic-concatN.lua')
    model_list['hg-generic-bypass']       = paths.dofile('hg-generic-bypass.lua')
    model_list['hg-generic-bypassdense']  = paths.dofile('hg-generic-bypassdense.lua')
    model_list['hg-generic-concatdense']  = paths.dofile('hg-generic-concatdense.lua')

    -- Ensemble nets
    model_list['hg-generic-ensemble']    = paths.dofile('hg-generic-ensemble.lua')
    model_list['hg-generic-ensemblev1']  = paths.dofile('hg-generic-ensemblev1.lua')
    model_list['hg-generic-ensemblev2']  = paths.dofile('hg-generic-ensemblev2.lua')
    model_list['hg-generic-ensemblev3']  = paths.dofile('hg-generic-ensemblev3.lua')
    model_list['hg-generic-ensemblev4']  = paths.dofile('hg-generic-ensemblev4.lua')
    model_list['hg-generic-ensemblev5']  = paths.dofile('hg-generic-ensemblev5.lua')
    model_list['hg-generic-ensemblev6']  = paths.dofile('hg-generic-ensemblev6.lua')

    -- SML (Small-Medium-Large) net
    model_list['sml_v1']    = paths.dofile('SML_v1.lua')
    model_list['sml_v2']    = paths.dofile('SML_v2.lua')
    model_list['sml_v3']    = paths.dofile('SML_v3.lua')
    model_list['sml_v3_1']  = paths.dofile('SML_v3_1.lua')
    model_list['sml_v3_2']  = paths.dofile('SML_v3_2.lua')
    model_list['sml_v3_3']  = paths.dofile('SML_v3_3.lua')
    model_list['sml_v3_4']  = paths.dofile('SML_v3_4.lua')
    model_list['sml_v3_5']  = paths.dofile('SML_v3_5.lua')
    model_list['sml_v3_6']  = paths.dofile('SML_v3_6.lua')
    model_list['sml_v4']    = paths.dofile('SML_v4.lua')
    model_list['sml_v5']    = paths.dofile('SML_v5.lua')

end

return additional_network_architectures