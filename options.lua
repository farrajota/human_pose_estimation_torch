--[[
    Options configurations for the train/test/benchmark scripts.
]]

projectDir = projectDir or './'

local function Parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-expID',        'hg-generic', 'Experiment ID')
    cmd:option('-ensembleID',   'hg-generic', 'Experiment ID')
    cmd:option('-dataset',        'mpii+lsp', 'Dataset choice: mpii | flic | lsp | coco | mpii+lsp')
    cmd:option('-data_dir',       'none', 'Path to store the dataset\'s data files.')
    cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
    cmd:option('-manualSeed',          2, 'Manually set RNG seed')
    cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-nGPU',                1, 'Number of GPUs to use by default')
    cmd:option('-nThreads',            2, 'Number of data loading threads')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-netType',  'hg-generic', 'Options: hg-stacked | hg-generic | hg-generic-rnn | hg-generic-wide | hg-generic-deconv | ' ..
                                                   'hg-generic-maxpool | hg-generic-inception | hg-generic-highres')
    cmd:option('-loadModel',      'none', 'Provide full path to a previously trained model')
    cmd:option('-continue',      'false', 'Pick up where an experiment left off')
    cmd:option('-branch',         'none', 'Provide a parent expID to branch off')
    cmd:option('-snapshot',           10, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-saveBest',       'true', 'Saves a snapshot of the model with the highest accuracy.')
    cmd:option('-task',       'pose-int', 'Network task: pose | pose-int')
    cmd:option('-nFeats',            256, 'Number of features in the hourglass (for hg-generic)')
    cmd:option('-nStack',              8, 'Number of stacks in the provided hourglass model (for hg-generic)')
    cmd:option('-genGraph',            1, 'Generate a graph of the network and save it to disk. 1 - Generate graph. 0 - Skip graph generation.')
    cmd:option('-clear_buffers', 'false', 'Empty network\'s buffers (gradInput, etc.) before saving the network to disk (if true).')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-optMethod',      'adam', 'Optimization method: rmsprop | sgd | nag | adadelta')
    cmd:option('-LR',             2.5e-4, 'Learning rate')
    cmd:option('-LRdecay',             0, 'Learning rate decay')
    cmd:option('-momentum',            0, 'Momentum')
    cmd:option('-weightDecay',         0, 'Weight decay')
    cmd:option('-beta1',              .9, 'adam: first moment coefficient')
    cmd:option('-beta2',            .999, 'adam: second moment coefficient')
    cmd:option('-alpha',             .99, 'rmsprop: Alpha - smoothing constant')
    cmd:option('-epsilon',          1e-8, 'adam/rmsprop: Epsilon')
    cmd:option('-crit',            'MSE', 'Criterion type: MSE, SmoothL1.')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-trainIters',      1000, 'Number of train batches/iterations per epoch')
    cmd:option('-testIters',       1000, 'Number of test batches/iterations used for accuracy validation per epoch')
    cmd:option('-nEpochs',          100, 'Total number of epochs to run')
    cmd:option('-batchSize',          2, 'Mini-batch size')
    cmd:option('-schedule', "{{50,2.5e-4,0},{15,1e-4,0},{10,5e-5,0}}", 'Optimization schedule. Overrides the previous configs if not empty.')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',          256, 'Input image resolution')
    cmd:option('-outputRes',          64, 'Output heatmap resolution')
    cmd:option('-scale',             .25, 'Degree of scale augmentation')
    cmd:option('-rotate',             30, 'Degree of rotation augmentation')
    cmd:option('-hmGauss',             1, 'Heatmap gaussian size')
    cmd:text()
    cmd:text(' ---------- Data augments ---------------------------------------')
    cmd:text()
    cmd:option('-colourNorm',     "false", 'mean/std norm')
    cmd:option('-colourjit',      "false", 'colour jittering (other method)')
    cmd:option('-centerjit',            0, 'Jitters the person\'s center position by N pixels in the (x,y) coords')
    cmd:option('-pca',            "false", 'ZCA whitening')
    cmd:option('-dropout',              0, 'dropout probability')
    cmd:option('-spatialdropout',       0, 'spatial dropout probability')
    cmd:option('-critweights',     "none", 'Apply (or not) different weights to the criterion: ' ..
                                           'linear- Linear | steep - steep linear | ' ..
                                           'log - Logaritmic | exp- Exponential | none - disabled')
    cmd:option('-rotRate',             .6, 'Rotation probability.')
    cmd:text()
    cmd:text(' ---------- Test options ---------------------------------------')
    cmd:text()
    cmd:option('-reprocess', "false",  'Utilize existing predictions from the model\'s folder.')
    cmd:option('-threshold',      .2, 'PCKh threshold (default 0.5)')
    cmd:option('-predictions',     0, 'Generate a predictions file (0-false | 1-true)')
    cmd:option('-plotSave',   "true", 'Save plot to file (true/false)')
    cmd:text()
    cmd:text(' ---------- Benchmark options --------------------------------------')
    cmd:text()
    cmd:option('-eval_plot_name', 'Ours', 'Plot the model with a specfied name.')
    cmd:text()
    cmd:text(' ---------- Demo options --------------------------------------')
    cmd:text()
    cmd:option('-demo_nsamples',            5, 'Number of samples to display predictions.')
    cmd:option('-demo_plot_save',     'false', 'Save plots to disk.')
    cmd:option('-demo_plot_networks_predictions', 'false', 'Plot the heatmap results of all network outputs.')
    cmd:text()


    local opt = cmd:parse(arg or {})
    opt.expDir = paths.concat(opt.expDir, opt.dataset)
    opt.save = paths.concat(opt.expDir, opt.expID)
    opt.ensemble = paths.concat(opt.expDir, opt.ensembleID)
    if opt.loadModel == '' or opt.loadModel == 'none' then
        opt.load = paths.concat(opt.save, 'model_final.t7')
        --opt.load = paths.concat(opt.save, 'best_model_accuracy.t7')
        --opt.load = paths.concat(opt.save, 'best_model_accu.t7')
    else
        opt.load = opt.loadModel
        opt.save = paths.dirname(opt.loadModel)
    end

    if not utils then
        utils = paths.dofile('util/utils.lua')
    end

    opt.schedule = utils.Str2TableFn(opt.schedule)

    if string.lower(opt.data_dir) == 'none' then
        opt.data_dir = ''
    end

    -- data augment testing vars
    opt.continue = utils.Str2Bool(opt.continue)
    opt.clear_buffers = utils.Str2Bool(opt.clear_buffers)
    opt.colourNorm  = utils.Str2Bool(opt.colourNorm)
    opt.colourjit   = utils.Str2Bool(opt.colourjit)
    opt.pca         = utils.Str2Bool(opt.pca)
    opt.saveBest    = utils.Str2Bool(opt.saveBest)
    --opt.critweights = utils.Str2Bool(opt.critweights)
    opt.reprocess = utils.Str2Bool(opt.reprocess)
    opt.demo_plot_save = utils.Str2Bool(opt.demo_plot_save)
    opt.demo_plot_networks_predictions = utils.Str2Bool(opt.demo_plot_networks_predictions)

    return opt
end

---------------------------------------------------------------------------------------------------

return {
  parse = Parse
}
