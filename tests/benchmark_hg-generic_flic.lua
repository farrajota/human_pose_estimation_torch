--[[
    Benchmark the 'hg-generic' network arch on the FLIC dataset.
]]


local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end


--[[ Benchmark network ]]
expID = 'hg-generic8'
eval_plot_name = 'hg-generic8'
exec_command(string.format('th benchmark.lua -dataset flic -expID %s -eval_plot_name %s',
                            expID, eval_plot_name))