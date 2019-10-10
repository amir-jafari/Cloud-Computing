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
-- Tables
a = {2, 4, 6, 8}
print (a[2])
a["b"] = 7
print(a.b)
print(a["b"])
a[2] = nil
print(a)

for k,v in pairs(a) do 
  print(k,v) 
end

-->=============================================================
-- More Options in Tables
t = {}             -- create an empty table
a = "var"
t[a] = 10        -- new entry
t[100] = "big"   -- new entry
print(t["var"])    
a1 = 100
print(t[a1])     
t["var"] = 2 * t["var"]    
print(t["var"])

-->=============================================================
-- read 10 lines storing them in a table
    n = {}
    for i=1,3 do
      print("enter a string")
      n[i] = io.read()
    end
  print(n)
-->=============================================================
-- print_r

lfs=require 'lfs'
lfs.chdir("/home/ajafari/Torch_/Class_Examples/")

print_R = require ("print_r")
print_R(a)

