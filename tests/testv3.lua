
-- 1 ensemble + dropout
os.execute('CUDA_VISIBLE_DEVICES=0 th train.lua -dataset flic -expID hg-generic8-ensemblev3 -netType hg-generic-ensemblev3 -nGPU 1 -optMethod adam -nThreads 4 -snapshot 0 -schedule "{{30,1e-3,0},{5,1e-4,0},{5,5e-5,0}}" -nStack 8 -genGraph 0')

-- 2 ensemble + spatial dropout
os.execute('CUDA_VISIBLE_DEVICES=0 th train.lua -dataset flic -expID hg-generic8-ensemblev4 -netType hg-generic-ensemblev4 -nGPU 1 -optMethod adam -nThreads 4 -snapshot 0 -schedule "{{30,1e-3,0},{5,1e-4,0},{5,5e-5,0}}" -nStack 8 -genGraph 0')

-- 3 ensemble + rrelu
os.execute('CUDA_VISIBLE_DEVICES=0 th train.lua -dataset flic -expID hg-generic8-ensemblev5 -netType hg-generic-ensemblev5 -nGPU 1 -optMethod adam -nThreads 4 -snapshot 0 -schedule "{{30,1e-3,0},{5,1e-4,0},{5,5e-5,0}}" -nStack 8 -genGraph 0')

-- 4 ensemble hg-generic 16
os.execute('CUDA_VISIBLE_DEVICES=0 th train.lua -dataset flic -expID hg-generic16-ensemblev6 -netType hg-generic-ensemblev6 -nGPU 1 -optMethod adam -nThreads 4 -snapshot 0 -schedule "{{30,1e-3,0},{5,1e-4,0},{5,5e-5,0}}" -nStack 16 -genGraph 0')

-- benchmarks
os.execute('th benchmark.lua -expID hg-generic8-ensemblev3 -threshold 0.2')
os.execute('th benchmark.lua -expID hg-generic8-ensemblev4 -threshold 0.2')
os.execute('th benchmark.lua -expID hg-generic8-ensemblev5 -threshold 0.2')
os.execute('th benchmark.lua -expID hg-generic8-ensemblev6 -threshold 0.2')


CUDA_VISIBLE_DEVICES=0 th train.lua -dataset flic -expID hg-generic16-ensemblev6 -netType hg-generic-ensemblev6 -nGPU 1 -optMethod adam -nThreads 4 -snapshot 0 -schedule "{{10,1e-3,0},{1,1e-4,0},{1,5e-5,0}}" -nStack 16 -genGraph 0