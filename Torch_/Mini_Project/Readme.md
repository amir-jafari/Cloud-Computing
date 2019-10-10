# Mini Project 1

## Guide

* Torch, MNIST data and dp package

Download train_mnist.lua file and answer all the questions on the MiniProject1.

train_mnist.lua

* Bug

There is bug in this code:

The code line from train_mnist.lua function classEval below:

```
function classEval(module, inputs, targets)
  cm:zero()
  for idx=1,inputs:size(1) do
    local input, target = inputs[idx], targets:narrow(1, idx, 1) 
    Tensor
    local output = module:forward(input)
    cm:add(output, target)
  end
 cm:updateValids()
 return cm.totalValid
end
```

Should be changed to the code below to fix issue:

```
function classEval(module, inputs, targets)
 cm:zero()
  for idx=1,inputs:size(1) do
    local input, target = inputs[idx], targets[idx]
    local output = module:forward(input)
    cm:add(output, target)
  end 
  cm:updateValids() 
  return cm.totalValid
end
```
