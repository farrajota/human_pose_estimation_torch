--[[
    Compute tests for the Test architecture experiments.
]]

local nstacks = 8

-- 1
print('***Test architecture experiment: 1/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-critweights-lin -threshold 0.2')
-- 2
print('***Test architecture experiment: 2/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-critweights-steep -threshold 0.2')
-- 3
print('***Test architecture experiment: 3/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-critweights-log -threshold 0.2')
-- 4
print('***Test architecture experiment: 4/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-critweights-exp -threshold 0.2')
-- 5
print('***Test architecture experiment: 5/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.1 -threshold 0.2')
-- 6
print('***Test architecture experiment: 6/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.2 -threshold 0.2')
-- 7
print('***Test architecture experiment: 7/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.3 -threshold 0.2')
-- 8
print('***Test architecture experiment: 8/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.4 -threshold 0.2')
-- 9
print('***Test architecture experiment: 9/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-spatialdropoutmiddle-0.5 -threshold 0.2')
-- 10
print('***Test architecture experiment: 10/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-concatv1 -threshold 0.2')
-- 11
print('***Test architecture experiment: 11/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-concatv2 -threshold 0.2')
-- 12
print('***Test architecture experiment: 12/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-rotate -threshold 0.2')
-- 13
print('***Test architecture experiment: 13/13***\n')
os.execute('th test.lua -expID hg-generic'..nstacks..'-scale -threshold 0.2')

