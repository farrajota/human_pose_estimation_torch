--[[
    Options configurations for the train script.
]]

local function Parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    --cmd:option('-expID',       'default-multigpu', 'Experiment ID')
    cmd:option('-expID',       'testes', 'Experiment ID')
    cmd:option('-dataset',        'flic', 'Dataset choice: mpii | flic')
    cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
    cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
    cmd:option('-manualSeed',          2, 'Manually set RNG seed')
    cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-nGPU',                1, 'Number of GPUs to use by default')
    cmd:option('-finalPredictions',    0, 'Generate a final set of predictions at the end of training (default no, set to 1 for yes)')
    cmd:option('-nThreads',            2, 'Number of data loading threads')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-netType',  'hg-stacked-wide', 'Options: hg-stacked | hg-generic | hg-generic-rnn | hg-generic-wide | hg-generic-deconv | ' ..
                                                   'hg-generic-maxpool | hg-generic-inception | hg-generic-highres')
    cmd:option('-loadModel',      'none', 'Provide full path to a previously trained model')
    cmd:option('-continue',        false, 'Pick up where an experiment left off')
    cmd:option('-branch',         'none', 'Provide a parent expID to branch off')
    cmd:option('-snapshot',            5, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-task',       'pose-int', 'Network task: pose | pose-int')
    cmd:option('-nFeats',            256, 'Number of features in the hourglass (for hg-generic)')
    cmd:option('-nStack',              2, 'Number of stacks in the provided hourglass model (for hg-generic)')
    cmd:option('-optimize',         true, 'Optimize network memory usage.')
    cmd:option('-genGraph',            1, 'Generate a graph of the network and save it to disk. 1 - Generate graph. 0 - Skip graph generation.')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-LR',             2.5e-4, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.0, 'Momentum')
    cmd:option('-weightDecay',       0.0, 'Weight decay')
    cmd:option('-crit',            'MSE', 'Criterion type: MSE, SmoothL1.')
    cmd:option('-optMethod',   'rmsprop', 'Optimization method: rmsprop | sgd | nag | adadelta')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-nEpochs',           10, 'Total number of epochs to run')
    cmd:option('-trainIters',       4000, 'Number of train iterations per epoch')
    cmd:option('-trainBatch',          4, 'Mini-batch size')
    cmd:option('-validIters',       2958, 'Number of validation iterations per epoch')
    cmd:option('-validBatch',          1, 'Mini-batch size for validation')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',          256, 'Input image resolution')
    cmd:option('-outputRes',          64, 'Output heatmap resolution')
    cmd:option('-trainFile',          '', 'Name of training data file')
    cmd:option('-validFile',          '', 'Name of validation file')
    cmd:option('-scaleFactor',       .25, 'Degree of scale augmentation')
    cmd:option('-rotFactor',          30, 'Degree of rotation augmentation')

    local opt = cmd:parse(arg or {})
    opt.expDir = paths.concat(opt.expDir, opt.dataset)
    opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
    opt.save = paths.concat(opt.expDir, opt.expID)
    
    if not (opt.validBatch >= opt.nGPU) then
      print('Converting validBatch from ' .. opt.validBatch .. ' to ' .. opt.nGPU)
      opt.validBatch = opt.nGPU
    end
    
    return opt
end

---------------------------------------------------------------------------------------------------

return {
  parse = Parse
}
