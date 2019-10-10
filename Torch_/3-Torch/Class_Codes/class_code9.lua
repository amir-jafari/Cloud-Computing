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
require 'cunn'
require 'cutorch'

start_time = os.time()

local model = nn.Sequential();  -- make a multi-layer perceptron
local R = 1; local S2 = 1; local S1 = 20; -- parameters
model:add(nn.Linear(R, S1))
model:add(nn.Tanh())
model:add(nn.Linear(S1, S2))

model:cuda()

local criterion = nn.MSECriterion()

criterion:cuda()

local Q = 21
local P = torch.Tensor(Q, R)
--local T = torch.ByteTensor(Q)

P[{{}, {1}}]=torch.range(-2,2,0.2)
P = P:cuda()
T = torch.sin(P)
T = T:cuda()

local params, gradParams = model:getParameters()

local optimState = {learningRate=0.01}

require 'optim'

for epoch=1,10000 do
  
  local function feval(params)
    gradParams:zero()

    local A = model:forward(P)
    local loss = criterion:forward(A, T)
    local dloss_doutput = criterion:backward(A, T)
    model:backward(P, dloss_doutput)

    return loss,gradParams
  end
  optim.sgd(feval, params, optimState)
end

end_time = os.time()
elapsed_time = os.difftime(end_time-start_time)
print('Elapsed time = ' .. elapsed_time)
require 'gnuplot'

local pp = torch.Tensor(201, 1)
local p2 = torch.range(-2,2,0.02)
pp[{{}, {1}}] = p2
local b = model:forward(pp)
local tt = torch.sin(pp)

