require "nn"

dataset={};
function dataset:size() return 500 end -- 100 examples
for i=1,dataset:size() do 
  local input = torch.randn(2);     -- normally distributed example in 2d
  local output = torch.Tensor(1);
  if input[1]*input[2]>0 then     -- calculate label for XOR function
    output[1] = -1;
  else
    output[1] = 1
  end
  dataset[i] = {input, output}
end



mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 20; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))


criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))