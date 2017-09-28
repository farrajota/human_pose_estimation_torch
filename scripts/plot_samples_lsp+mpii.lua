--[[
    Plot body joint heatmaps and skeletons of random images from the LSP (+MPII) dataset and save them do disk.
]]


configs = {
    dataset = 'lsp+mpii',
    expID = 'hg-generic8', --'final-best',
    demo_nsamples = 100,
    demo_plot_save = 'true',
    manualSeed = 2,
    demo_plot_screen = 'false',
    test_load_best = 'true',
}

--[[ concatenate options fields to a string ]]
local str_args = ''
for k, v in pairs(configs) do
    str_args = str_args .. ('-%s %s '):format(k, v)
end

--[[ Setup command ]]
command = 'qlua demo.lua ' .. str_args

--[[ Run command ]]
print('Executing command: ' .. command)
print()
os.execute(command)