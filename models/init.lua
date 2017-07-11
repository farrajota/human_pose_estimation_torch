--[[
    List of available models to load.
]]


local model_list = {}
paths.dofile('test/init.lua')(model_list)

model_list['hg-generic-best']     = paths.dofile('hg-generic-best.lua')  -- best combo of all models
model_list['hg-generic-ensemble'] = paths.dofile('hg-generic-ensemble.lua')  -- Ensemble net
model_list['hg-stacked']          = paths.dofile('hg-stacked.lua')
model_list['hg-stacked-no-int']   = paths.dofile('hg-stacked-no-int.lua')
model_list['hg-generic']          = paths.dofile('hg-generic.lua')
model_list['sml']                 = paths.dofile('SML.lua')  -- SML net

return model_list
