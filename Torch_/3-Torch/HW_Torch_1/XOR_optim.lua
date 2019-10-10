require 'nn'


model = nn.Sequential()  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 20 -- parameters
model:add(nn.Linear(inputs, HUs))
model:add(nn.Tanh())
model:add(nn.Linear(HUs, outputs))

criterion = nn.MSECriterion()


batchSize = 128
batchInputs = torch.DoubleTensor(batchSize, inputs) 
batchLabels = torch.DoubleTensor(batchSize)         

for i = 1, batchSize do
   local input = torch.randn(2)     -- normally distributed example in 2d
   local label
   if input[1] * input[2] > 0 then  -- calculate label for XOR function
      label = -1
   else
      label = 1
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
end

x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(model:forward(x))
x[1] =  0.5; x[2] = -0.5; print(model:forward(x))
x[1] = -0.5; x[2] =  0.5; print(model:forward(x))
x[1] = -0.5; x[2] = -0.5; print(model:forward(x))