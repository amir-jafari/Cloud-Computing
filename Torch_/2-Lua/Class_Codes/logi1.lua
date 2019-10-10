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
-- Modules Creation
local logi1 = {}

function logi1.a (x, y)
  print(x and y)
end

function logi1.o(x ,y)
  print(x or y)
end

return logi1
