--[[
    Plot body joint heatmaps and skeletons of random images from the LSP dataset and save them do disk.
]]


configs = {
    dataset = 'lsp',
    expID = 'final-best',
    demo_nsamples = 5,
    demo_plot_save = 'true',
    manualSeed = 2,
    demo_plot_screen = 'false',
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