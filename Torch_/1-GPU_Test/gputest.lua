-- Max Jaderberg 4/9/13
-- GPU Effectiveness test

--require('mobdebug').start()
require 'torch'
require 'sys'
require 'nn'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('GPU Benchmark. Max Jaderberg.')
cmd:text()
cmd:text('Options:')
cmd:option('-size', 24, 'size of images')
cmd:option('-N', 1000, 'number of images')
cmd:option('-threads', 2, 'number of threads')
cmd:text()
opt = cmd:parse(arg or {})

torch.setnumthreads(opt.threads)

torch.setdefaulttensortype("torch.FloatTensor")


-- Input size
sz = opt.size
finalsize = (sz - 4)/2 --for a 5x5 filter
Ntest = opt.N

-- Test input
x = torch.rand(Ntest, 3, sz, sz)

-- Simple network
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 64, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2))
model:add(nn.Reshape(64*finalsize*finalsize))
model:add(nn.Linear(64*finalsize*finalsize, 128))
model:add(nn.Tanh())
model:add(nn.Linear(128, 2))
model:add(nn.Tanh())
criterion = nn.MSECriterion()

-- Test ouput
y = torch.rand(Ntest, 2)


-- Test routine
local runtest = function() 
  print('==> Type is '..x:type())

	for i = 1,x:size(1) do
		--xlua.progress(i, x:size(1))

		local yp = model:forward(x[i])
		local err = criterion:forward(yp, y[i])
		local df_do = criterion:backward(yp, y[i])
		model:backward(x[i], df_do)
	end
end

-- CPU TEST
cputime0 = sys.clock()
runtest()
cputime1 = sys.clock()
cputime = cputime1 - cputime0
print('CPU Time: '.. (cputime*1000) .. 'ms')

-- GPU TEST
require 'cunn'
x = x:cuda()
y = y:cuda()
model:cuda()
criterion:cuda()

gputime0 = sys.clock()
runtest()
gputime1 = sys.clock()
gputime = gputime1 - gputime0
print('GPU Time: '.. (gputime*1000) .. 'ms')

print('------------------')
print('GPU speedup: '..cputime/gputime..'x')


