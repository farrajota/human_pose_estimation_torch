--[[
    Benchmark the 'hg-stacked' network arch on the LSP(+MPII) dataset.
]]


local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end


--[[ Benchmark network ]]
expID = 'hg-stacked'
eval_plot_name = 'hg-stacked'
exec_command(string.format('th benchmark.lua -dataset lsp+mpii -expID %s -eval_plot_name %s',
                            expID, eval_plot_name))