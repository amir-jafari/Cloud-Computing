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
-- Functions

function times(x)
  return function (y) return x * y end
end
f = times(3)
print(f(2))


function f(y,q) 
  return y*y, 1/q
end

z,w = f(4,2)
print(z,w)

a = {1, 2, 3, 4, 5, 6, 7, 8, 9}
x = 0
for i,v in ipairs(a) do
  x = x + v
end
print(x)
a["z"] = 7
print(a.z)

-->=============================================================
-- Functions: whos.lua, by michal-kottman
-- All global variables in Lua reside in a table available as global variable _G (yes, _G._G == _G). Therefore if you want to list all global variable, you can iterate over the table using pairs()

local base = {}
for k,v in pairs(_G) do
    base[k] = true
end
return function()
    for k,v in pairs(_G) do
        if not base[k] then print(k, type(v), v) end
    end
end