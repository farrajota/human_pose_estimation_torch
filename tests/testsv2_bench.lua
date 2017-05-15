--[[
    Compute benchmarks for the test architectures.
]]

local nstacks = 8

-- 1
print('***Benchmark 1/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-critweights-lin -threshold 0.2')
-- 2
print('***Benchmark 2/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-critweights-steep -threshold 0.2')
-- 3
print('***Benchmark 3/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-critweights-log -threshold 0.2')
-- 4
print('***Benchmark 4/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-critweights-exp -threshold 0.2')
-- 5
print('***Benchmark 5/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.1 -threshold 0.2')
-- 6
print('***Benchmark 6/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.2 -threshold 0.2')
-- 7
print('***Benchmark 7/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.3 -threshold 0.2')
-- 8
print('***Benchmark 8/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.4 -threshold 0.2')
-- 9
print('***Benchmark 9/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.5 -threshold 0.2')
-- 10
print('***Benchmark 10/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-concatv1 -threshold 0.2')
-- 11
print('***Benchmark 11/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-concatv2 -threshold 0.2')
-- 12
print('***Benchmark 12/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-rotate -threshold 0.2')
-- 13
print('***Benchmark 13/13***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-scale -threshold 0.2')

