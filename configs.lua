--[[
    Loads necessary libraries and files.
]]


require 'paths'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
--require 'nnx'
require 'nngraph'
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('projectdir.lua') -- Project directory
paths.dofile('data.lua')
paths.dofile('sample_batch.lua')
paths.dofile('models/modules/NoBackprop.lua')

paths.dofile('util/eval.lua')
paths.dofile('util/Logger.lua')
paths.dofile('util/store.lua')
paths.dofile('util/draw.lua')
utils = paths.dofile('util/utils.lua')


-----------------------------------------------------------
-- Load options
-----------------------------------------------------------

if not opt then
    local opts = paths.dofile('options.lua')
    opt = opts.parse(arg)

    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)

    if opt.GPU >= 1 then
        cutorch.setDevice(opt.GPU)
    end

    if opt.branch ~= 'none' or opt.continue then
        -- Continuing training from a prior experiment
        -- Figure out which new options have been set

        if paths.filep(opt.save .. '/options.t7') then
            local setOpts = {}
            for i = 1,#arg do
                if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
            end
            -- Where to load the previous options/model from
            if opt.branch ~= 'none' then
                opt.load = opt.expDir .. '/' .. opt.branch
            else
                opt.load = opt.expDir .. '/' .. opt.expID
            end

            -- Keep previous options, except those that were manually set
            local opt_ = opt
            opt = torch.load(opt_.load .. '/options.t7')
            opt.save = opt_.save
            opt.load = opt_.load
            opt.continue = opt_.continue
            for i = 1,#setOpts do
                opt[setOpts[i]] = opt_[setOpts[i]]
            end

            -- determine highest epoc and load corresponding model
            local last_epoch = torch.load(opt.load .. '/last_epoch.t7')
            epoch = last_epoch

            -- If there's a previous optimState, load that too
            --if paths.filep(opt.load .. '/optim.t7') then
            --    optimState = torch.load(opt.load .. '/optim.t7')
            --    optimState.learningRate = opt.LR
            --elseif paths.filep(opt.load .. '/optim_' .. last_epoch .. '.t7') then
            --    optimState = torch.load(opt.load .. '/optim_' .. last_epoch .. '.t7')
            --end
        else
            epoch = 1
        end
    else
        epoch = 1
    end

    opt.epochNumber = epoch
    nEpochs = opt.nEpochs

    -- Training hyperparameters
    if not optimState then
        if type(opt.schedule)=='table' then
            local schedule = {}
            local schedule_id = 0
            for i=1, #opt.schedule do
                table.insert(schedule, {schedule_id+1, schedule_id+opt.schedule[i][1],
                    opt.schedule[i][2],
                    opt.schedule[i][3]})
                schedule_id = schedule_id+opt.schedule[i][1]
            end
            optimStateFn = function(epoch)
                for k, v in pairs(schedule) do
                    if v[1] <= epoch and v[2] >= epoch then
                        return {
                            learningRate = v[3],
                            learningRateDecay = opt.LRdecay,
                            momentum = opt.momentum,
                            dampening = 0.0,
                            weightDecay = v[4],
                            beta1 = opt.beta1,  -- adam
                            beta2 = opt.beta2,  -- adam
                            alpha = opt.alpha,  -- rmsprop
                            epsilon = opt.epsilon,  -- adam/rmsprop
                            end_schedule = (v[2]==epoch and 1) or 0
                        }
                    end
                end
                return optimState
            end
            -- determine the maximum number of epochs
            for k, v in pairs(schedule) do
                nEpochs = math.min(v[2])
            end
        else
            optimStateFn = function(epoch)
                return {
                    learningRate = opt.LR,
                    learningRateDecay = opt.LRdecay,
                    momentum = opt.momentum,
                    dampening = 0.0,
                    weightDecay = opt.weightDecay,
                    beta1 = opt.beta1,  -- adam
                    beta2 = opt.beta2,  -- adam
                    alpha = opt.alpha,  -- rmsprop
                    epsilon = opt.epsilon,  -- adam/rmsprop
                }
            end
        end
    end

    -- Random number seed
    if opt.manualSeed ~= -1 then
        torch.manualSeed(opt.manualSeed)
    else
        torch.seed()
    end

    -- Save options to experiment directory
    torch.save(opt.save .. '/options.t7', opt)
end


-----------------------------------------------------------
-- Functions
-----------------------------------------------------------

local function get_num_keypoints(outputDim)
    if outputDim then
        return outputDim
    else
        local data_loader = select_dataset_loader(opt.dataset, 'train')
        local nJoints = data_loader.train.num_keypoints
        return {nJoints, opt.outputRes, opt.outputRes}
    end
end

------------------------------------------------------------------------------------------------------------

function load_model(mode)
    local str = string.lower(mode)
    if str == 'train' then
        outputDim = get_num_keypoints(outputDim)
        if opt.netType == 'hg-stacked' then
            if opt.task == 'pose-int' then
                local newDim = {}
                newDim[1] = outputDim
                newDim[2] = outputDim
                outputDim = newDim
                opt.nOutputs = 2
            else
                opt.nOutputs = 1
            end
        else
            local newDim = {}
            for i=1, opt.nStack do
                table.insert(newDim, outputDim)
            end
            outputDim = newDim
            opt.nOutputs = opt.nStack
        end

        -- Load model
        model, criterion = paths.dofile('model.lua')

        model:training()
    elseif str == 'test' then
        -- load model
        model = torch.load(opt.load)

        if opt.GPU >= 1 then
            opt.dataType = 'torch.CudaTensor'  -- Use GPU
            model:cuda()

            -- convert to cuda
            if pcall(require, 'cudnn') then
                print('Converting model backend to cudnn...')
                cudnn.convert(model, cudnn):cuda()
                print('Done.')
            end

            model = utils.loadDataParallel(model, 1) -- load model into 1 GPU
            model:cuda()
        else
            opt.dataType = 'torch.FloatTensor' -- Use CPU
            model:float()
        end

        model:evaluate()
    else
        error(('Invalid mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

function process_mean_std()
    if opt.colourNorm then
        print('Loading mean/std normalization values... ')
        local fname_meanstd = paths.concat(opt.expDir, 'meanstd_cache.t7')

        if paths.filep(fname_meanstd) then
            -- load mean/std from disk
            print('Loading mean/std cache from disk: ' .. fname_meanstd)
            opt.meanstd = torch.load(fname_meanstd, meanstd)
        else
            -- compute mean/std
            local data_loader = select_dataset_loader(opt.dataset, 'train')
            print('mean/std cache file not found. Computing mean/std for the ' .. opt.dataset ..' dataset:')
            local meanstd = ComputeMeanStd(data_loader.train)
            print('Saving mean/std cache to disk: ' .. fname_meanstd)
            torch.save(fname_meanstd, meanstd)
            opt.meanstd = meanstd
        end
        --print(opt.meanstd)
    end
end