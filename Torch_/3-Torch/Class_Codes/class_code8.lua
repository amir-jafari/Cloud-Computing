-- %%%%%%%%%%%%% Deep Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%
-- %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- Dr. Martin Hagan----->Email: mhagan@okstate.edu 
-- Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
-- %%%%%%%%%%%%% Date:
-- V1 Jan - 01 - 2017
-- V2 Sep - 29 - 2017
-- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- %%%%%%%%%%%%% Torch Examples %%%%%%%%%%%%%%%%%%%%%%%%%%%%
-->=============================================================
require 'nn'


local model = nn.Sequential();  -- make a multi-layer perceptron
local R = 1; local S2 = 1; local S1 = 10; -- parameters
model:add(nn.Linear(R, S1))
model:add(nn.Tanh())
model:add(nn.Linear(S1, S2))


local criterion = nn.MSECriterion()


local Q = 21
local P = torch.Tensor(Q, R)
Epoch = 500
perf=torch.Tensor(Epoch,1)
P[{{}, {1}}]=torch.range(-2,2,0.2)
T = torch.sin(P)

local function runtest()
  local params, gradParams = model:getParameters()

  local optimState = {learningRate=0.1}  

  require 'optim'

  for epoch=1,Epoch do
  
    local function feval(params)
    gradParams:zero()

      local A = model:forward(P)
      local loss = criterion:forward(A, T)
      local dloss_doutput = criterion:backward(A, T)
      model:backward(P, dloss_doutput)

      return loss,gradParams
    end
  optim.sgd(feval, params, optimState)
  local A1 = model:forward(P)
  perf[epoch] = criterion:forward(A1, T)
  end
  
end
-- CPU TEST
require 'sys'
cputime0 = sys.clock()
runtest()
cputime1 = sys.clock()
cputime = cputime1 - cputime0
print('CPU Time: '.. (cputime*1000) .. 'ms')


-- GPU TEST
require 'cunn'
P = P:cuda()
T = T:cuda()

model:cuda()
criterion:cuda()

gputime0 = sys.clock()
runtest()
gputime1 = sys.clock()
gputime = gputime1 - gputime0
print('GPU Time: '.. (gputime*1000) .. 'ms')
 
 
 
 
print('------------------')
print('GPU speedup: '..cputime/gputime..'x')
