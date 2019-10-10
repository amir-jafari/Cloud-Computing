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
--Tensor
--- creation of a 4D-tensor 4x5x6x2
require 'torch'
z = torch.Tensor(4,5,6,2)
--- for more dimensions, (here a 6D tensor) one can do:
s = torch.LongStorage(6)
s[1] = 4; s[2] = 5; s[3] = 6; s[4] = 2; s[5] = 7; s[6] = 3;

print(s)

x = torch.Tensor(s)

print(x)
print(x:nDimension())
print(x:size())



x = torch.Tensor(7,7,7)
print(x[3][4][5])


x = torch.Tensor(4,5)
s = x:storage()

for i=1,s:size() do -- fill up the Storage
  s[i] = i
end

x = torch.Tensor(4,5)
i = 0

x:apply(function() i = i + 1 return i end)

x = torch.Tensor(5):zero()

x:narrow(1, 2, 3):fill(1)


y = torch.Tensor(x:size()):copy(x)
y = x:clone()


x = torch.Tensor(2,5):fill(3.14)
y = torch.Tensor(x)
y:zero()


i = 0
x = torch.Tensor(5):apply(function(x)
  i = i + 1
  return i
end)
y = x:clone()
y:fill(1)


x = torch.Tensor(2,3):fill(1)
y = x:contiguous():fill(2)

z = x:t():contiguous():fill(3.14)




torch.isTensor(torch.randn(3,4))
torch.isTensor(torch.randn(3,4)[1])
torch.isTensor(torch.randn(3,4)[1][2])


x = torch.Tensor(4,5)
x:nDimension()


x = torch.Tensor(4,5):zero()
x:size(2)



x = torch.Tensor(4,5)
s = x:storage()
for i=1,s:size() do -- fill up the Storage
  s[i] = i
end


x = torch.Tensor(3,3)
i = 0; x:apply(function() i = i + 1; return i end)

print(x[2])

print(x[2][3])

print(x[{2,3}])

x = torch.Tensor(2,5):fill(3.14)
y = torch.Tensor():set(x)
y:zero()


x = torch.Tensor(5, 6):zero()
y = x:narrow(1, 2, 3)
y:fill(1)
print(y)
print(x)

x = torch.Tensor(5, 6):zero()
y = x:sub(2,4):fill(1) 
print(y)
print(x)
z = x:sub(2,4,3,4):fill(2) 
print(y)
print(x)

x = torch.Tensor(5,6):zero()
y = x:select(1, 2):fill(2)
print(y)
print(x)
z = x:select(2,5):fill(5)


x = torch.Tensor(5, 6):zero()
x[{ 1,3 }] = 1
x[{ 2,{2,4} }] = 2 
x[{ {},4 }] = -1
x[{ {},2 }] = torch.range(1,5) 
x[torch.lt(x,0)] = -2



