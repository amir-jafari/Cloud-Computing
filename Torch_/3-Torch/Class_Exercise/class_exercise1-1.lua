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
--XOR Optim
require 'nn'


model = nn.Sequential()  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 20 -- parameters
model:add(nn.Linear(inputs, HUs))
model:add(nn.Tanh())
model:add(nn.Linear(HUs, 2))
model:add(nn.LogSoftMax())

criterion = nn.MSECriterion()


batchSize = 128
batchInputs = torch.DoubleTensor(batchSize, inputs) 
batchLabels = torch.DoubleTensor(batchSize,inputs) 
perf = torch.DoubleTensor(5000)

-->############################################################################################################
-- Question 1:
-->=============================================================
--a) Modify the follwing for loop that the we have labes 1 and zero
--   for the following inputs. Explain Why we need to have 2 targets.
-->=============================================================

for i = 1, batchSize do
   local input = torch.randn(2)     -- normally distributed example in 2d
   local label = torch.Tensor(2)
   if input[1] * input[2] > 0 then  -- calculate label for XOR function
-->  ?
-->  ?
   else
-->  ?
-->  ?
   end
   batchInputs[i]:copy(input)
   batchLabels[i] = label
end


-- Put parameters into vector
params, gradParams = model:getParameters()

local optimState = {learningRate = 0.01}

require 'optim'

for epoch = 1, 5000 do
   -- local function we give to optim
   -- it takes current weights as input, and outputs the loss
   -- and the gradient of the loss with respect to the weights
   -- gradParams is calculated implicitly by calling 'backward',
   -- because the model's weight and bias gradient tensors
   -- are simply views onto gradParams
   function feval(params)
      gradParams:zero()

      local outputs = model:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutputs = criterion:backward(outputs, batchLabels)
      model:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
   end
   optim.sgd(feval, params, optimState)
   local outputsa = model:forward(batchInputs)
   perf[epoch] = criterion:forward(outputsa, batchLabels)
end



-->############################################################################################################
-- Question 2:
-->=============================================================
--a) Use require gnuplot and plot the mean square error.
--    Change for different number epochs and explain your results.
-->=============================================================











-->############################################################################################################
-- Question 3:
-->=============================================================
--a) What are the results for the following inputs. Simulate the network
--   using trained netowrk and check out the results. 
-->=============================================================


x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; 
x[1] =  0.5; x[2] = -0.5; 
x[1] = -0.5; x[2] =  0.5; 
x[1] = -0.5; x[2] = -0.5; 
