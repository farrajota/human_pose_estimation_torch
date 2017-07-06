--[[
    Benchmark the 'best' network arch on the LSP dataset.
]]


local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end


--[[ Benchmark network ]]
expID = 'final-best'
eval_plot_name = '\'Ours-best\''
exec_command(string.format('th benchmark.lua -dataset lsp -expID %s -eval_plot_name %s',
                            expID, eval_plot_name))

--[[ Benchmark ensemble network ]]
expID = 'final-ensemble-best'
eval_plot_name = '\'Ours-ensemble-best\''
exec_command(string.format('th benchmark.lua -dataset lsp -expID %s -eval_plot_name %s',
                            expID, eval_plot_name))