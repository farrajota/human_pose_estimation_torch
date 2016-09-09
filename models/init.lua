--[[
    List of available models for Flic/MPII.
]]

local model_list = {}

-- stacked
model_list['hg-stacked']        = paths.dofile('hg-stacked.lua')
model_list['hg-stacked-no-int'] = paths.dofile('hg-stacked-no-int.lua')
-- generic
model_list['hg-generic'] = paths.dofile('hg-generic.lua')
-- modified generic
model_list['hg-generic-deconv']     = paths.dofile('hg-generic-deconv.lua')
model_list['hg-generic-maxpool']    = paths.dofile('hg-generic-maxpool.lua')
model_list['hg-generic-deception']  = paths.dofile('hg-generic-deception.lua')
model_list['hg-generic-inception']  = paths.dofile('hg-generic-inception.lua')
model_list['hg-generic-wide']       = paths.dofile('hg-generic-wide.lua')
model_list['hg-generic-highres']    = paths.dofile('hg-generic-highres.lua')
-- rnn generic
model_list['hg-generic-rnn']            = paths.dofile('hg-generic-rnn.lua')
model_list['hg-generic-rnn-deception']  = paths.dofile('hg-generic-rnn-deception.lua')
model_list['hg-generic-rnn-inception']  = paths.dofile('hg-generic-rnn-deception.lua')
model_list['hg-generic-rnn-maxpool']    = paths.dofile('hg-generic-rnn-maxpool.lua')
model_list['hg-generic-rnn-twin']       = paths.dofile('hg-generic-rnn-twin.lua')


return model_list
