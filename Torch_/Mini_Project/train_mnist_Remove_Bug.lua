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


require 'dp'

-- Load the mnist data set
ds = dp.Mnist()

-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw')
trainTargets = ds:get('train', 'targets', 'b')
validInputs = ds:get('valid', 'inputs', 'bchw')
validTargets = ds:get('valid', 'targets', 'b')
testInputs = ds:get('test', 'inputs', 'bchw')
testTargets = ds:get('test', 'targets', 'b')

-- Create a two-layer network
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax()) 

-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion()

require 'optim'
-- allocate a confusion matrix
cm = optim.ConfusionMatrix(10)
-- create a function to compute 
function classEval(module, inputs, targets)
   cm:zero()
   for idx=1,inputs:size(1) do
      local input, target = inputs[idx], targets:narrow(1,idx,1)
      local output = module:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end

 require 'dpnn'
function trainEpoch(module, criterion, inputs, targets)
   for i=1,inputs:size(1) do
      local idx = math.random(1,inputs:size(1))
      local input, target = inputs[idx], targets[idx] --targets:narrow(1,idx,1)
      -- forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      -- backward
      local gradOutput = criterion:backward(output, target)
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput)
      -- update
      module:updateGradParameters(0.9) -- momentum (dpnn)
      module:updateParameters(0.1) -- W = W - 0.1*dL/dW
   end
end

 bestAccuracy, bestEpoch = 0, 0
wait = 0
for epoch=1,30 do
   trainEpoch(module, criterion, trainInputs, trainTargets)
   local validAccuracy = classEval(module, validInputs, validTargets)
   if validAccuracy > bestAccuracy then
      bestAccuracy, bestEpoch = validAccuracy, epoch
      --torch.save("/path/to/saved/model.t7", module)
      print(string.format("New maxima : %f @ %f", bestAccuracy, bestEpoch))
      wait = 0
   else
      wait = wait + 1
      if wait > 30 then break end
   end
end
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))