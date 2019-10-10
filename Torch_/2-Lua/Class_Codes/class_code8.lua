-- %%%%%%%%%%%%% Deep Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%
-- %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- Dr. Martin Hagan----->Email: mhagan@okstate.edu 
-- Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
-- %%%%%%%%%%%%% Date:
-- V1 Jan - 01 - 2017
-- V2 Sep - 29 - 2017
-- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- %%%%%%%%%%%%% Lua Examples %%%%%%%%%%%%%%%%%%%%%%%%%%%%

-->=============================================================
-- Array

a = {}   
for i=1, 1000 do
  a[i] = 0
end

-->=============================================================
-- Matrix 1
R= 2
C =2

mat1 = {}          
for i=1,R do
  mat1[i] = {}     
  for j=1,C do
    mat1[i][j] = 2
  end
end
print(mat1)

for k,v in pairs(mat1[1]) do print(k,v) end
-->=============================================================
-- Matrix 2
R = 2
C = 1

mat2 = {} 

for i=1,R do
  for j=1,C do
    mat2[i*C + j] = 1
  end
end

for k1,v1 in pairs(mat2) do print(k1,v1) end
-->=============================================================
-- Unpack
A={{1,2},{3,4}}
A[1][1]=6

for i=1,#A do
print(unpack(A[i]))
end