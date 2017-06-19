--[[
    Compute accuracy evaluation for the experimental architectures.
]]

local nstacks = 8

-- 1
print('***Test architecture experiment: 1/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..' -threshold 0.2')
-- 2
print('***Test architecture experiment: 2/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-meanstd -threshold 0.2')
-- 3
print('***Test architecture experiment: 3/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-colourjit -threshold 0.2')
-- 4
print('***Test architecture experiment: 4/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-highres -inputRes 512 -threshold 0.2')
-- 5
print('***Test architecture experiment: 5/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-centerjit -threshold 0.2')
-- 6
print('***Test architecture experiment: 6/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-dropoutbefore -threshold 0.2')
-- 7
print('***Test architecture experiment: 7/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-dropoutmiddle -threshold 0.2')
-- 8
print('***Test architecture experiment: 8/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutbefore -threshold 0.2')
-- 9
print('***Test architecture experiment: 9/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle -threshold 0.2')
-- 10
print('***Test architecture experiment: 10/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-critweights -threshold 0.2')
-- 11
print('***Test architecture experiment: 11/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-concatv1 -threshold 0.2')
-- 12
print('***Test architecture experiment: 12/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-concatv2 -threshold 0.2')
-- 13
print('***Test architecture experiment: 13/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-concatdense -threshold 0.2')
-- 14
print('***Test architecture experiment: 14/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-bypass -threshold 0.2')
-- 15
print('***Test architecture experiment: 15/15***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-bypassdense -threshold 0.2')
