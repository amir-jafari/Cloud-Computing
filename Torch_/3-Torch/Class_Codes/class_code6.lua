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
--XOR Data

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
