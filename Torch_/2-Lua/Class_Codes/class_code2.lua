-- %%%%%%%%%%%%% Deep Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%
-- %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- Dr. Martin Hagan----->Email: mhagan@okstate.edu 
-- Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
-- %%%%%%%%%%%%% Date:
-- V1 Jan - 01 - 2017
-- V2 Sep - 29 - 2017
-- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- %%%%%%%%%%%%% Lua Examples %%%%%%%%%%%%%%%%%%%%%%%%%%%%
-- loops in Lua

x = 0
    local i = 1
a = {1,2,3,4,5,6,7,8,9,10}
    while a[i] do
      x = x + a[i]
      i = i + 1
      print(x)
      print(i)
    end
    
print(x)
print(i)

-- Repeat
local i = 1
x = 0
  
repeat
  x = x +a[i]
  i = i + 1
until i==5

print(x)
print(i)
