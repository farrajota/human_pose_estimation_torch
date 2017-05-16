--[[
    Logger functions for easier logging data management.
]]

if not utils then utils = paths.dofile('utils.lua') end

------------------------------------------------------------------------------------------------------------

local function copy_parameters_models(modelA, modelB)
--[[ Copy parameters (bias/weights) from a network A to B. ]]
    local paramsA = modelA:parameters()
    local paramsB = modelB:parameters()
    for i=1, #paramsA do
        paramsB[i]:copy(paramsA[i])
    end
end

------------------------------------------------------------------------------------------------------------

local function store(model, modelSave, optimState, epoch, opt, flag)
    local filename_model, filename_optimstate
    if flag then
        filename_model = paths.concat(opt.save,'model_' .. epoch ..'.t7')
        filename_optimstate = paths.concat(opt.save,'optim_' .. epoch ..'.t7')
    else
        filename_model = paths.concat(opt.save,'model.t7')
        filename_optimstate = paths.concat(opt.save,'optim.t7')
    end

    -- copy parameters from a model to another
    copy_parameters_models(model, modelSave)

    print('Saving model snapshot to: ' .. filename_model)
    torch.save(filename_optimstate, optimState)
    torch.save(filename_model, modelSave)
    torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)

    -- make a symlink to the last trained model
    local filename_symlink = paths.concat(opt.save,'model_final.t7')
    if paths.filep(filename_symlink) then
        os.execute(('rm %s'):format(filename_symlink))
    end
    os.execute(('ln -s %s %s'):format(filename_model, filename_symlink))
end

------------------------------------------------------------------------------------------------------------

function storeModel(model, modelSave, optimState, epoch, opt)
--[[ store model snapshot ]]

    if opt.snapshot > 0 then
        if epoch%opt.snapshot == 0 then
           store(model, modelSave, optimState, epoch, opt, true)
        end
    elseif opt.snapshot < 0 then
        if epoch%math.abs(opt.snapshot) == 0 then
           store(model, modelSave, optimState, epoch, opt, false)
        end
    else
        -- save only at the last epoch
        if epoch == opt.nEpochs then
          store(model, modelSave, optimState, epoch, opt, false)
        end
    end
end

------------------------------------------------------------------------------------------------------------

function storeModelBest(model, modelSave, opt)
--[[ store model snapshot of the best model]]

    local filename_model = paths.concat(opt.save,'best_model_accuracy.t7')

  -- copy parameters from a model to another
    copy_parameters_models(model, modelSave)

    print('New best accuracy detected! Saving model snapshot to disk: ' .. filename_model)

    torch.save(filename_model, modelSave)
end

------------------------------------------------------------------------------------------------------------

--[[
local function store_old(model, optimState, epoch, flag)
   if flag then
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model_' .. epoch ..'.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
      --torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:clearState())

      torch.save(paths.concat(opt.save,'optim_' .. epoch ..'.t7'), optimState)
      torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)

   else
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model.t7'), model)
      --torch.save(paths.concat(opt.save, 'model.t7'), model:clearState())

      torch.save(paths.concat(opt.save,'optim.t7'), optimState)
      torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)
   end

   collectgarbage()
end

------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------

function storeModel_old(model, optimState, epoch, opt)
   -- store model snapshot
   if opt.snapshot > 0 then
     if epoch%opt.snapshot == 0 then
        store(model, optimState, epoch, true)
     end

   elseif opt.snapshot < 0 then
     if epoch%math.abs(opt.snapshot) == 0 then
        store(model, optimState, epoch, false)
     end
   else
      -- save only at the last epoch
      if epoch == opt.nEpochs then
        store(model, optimState, epoch, false)
      end
   end
end
]]