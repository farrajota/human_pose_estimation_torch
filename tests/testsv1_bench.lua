--[[
    Compute benchmarks for the test architectures.
]]

local nstacks = 8

-- 1
print('***Benchmark 1/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..' -threshold 0.2')
-- 2
print('***Benchmark 2/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-meanstd -threshold 0.2')
-- 3
print('***Benchmark 3/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-colourjit -threshold 0.2')
-- 4
print('***Benchmark 4/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-highres -inputRes 512 -threshold 0.2')
-- 5
print('***Benchmark 5/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-centerjit -threshold 0.2')
-- 6
print('***Benchmark 6/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-dropoutbefore -threshold 0.2')
-- 7
print('***Benchmark 7/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-dropoutmiddle -threshold 0.2')
-- 8
print('***Benchmark 8/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutbefore -threshold 0.2')
-- 9
print('***Benchmark 9/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle -threshold 0.2')
-- 10
print('***Benchmark 10/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-critweights -threshold 0.2')
-- 11
print('***Benchmark 11/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-concatv1 -threshold 0.2')
-- 12
print('***Benchmark 12/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-concatv2 -threshold 0.2')
-- 13
print('***Benchmark 13/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-concatdense -threshold 0.2')
-- 14
print('***Benchmark 14/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-bypass -threshold 0.2')
-- 15
print('***Benchmark 15/15***\n')
os.execute('th benchmark.lua -expID hg-generic'..nstacks..'-bypassdense -threshold 0.2')
