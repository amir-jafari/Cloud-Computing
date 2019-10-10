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
-- golbal variable

a = 2
b = 3

c = a +b

print(c)

-->=============================================================
-- local varibale
local a = 2
local b = 3

local c = a +b

print(c)

-->=============================================================
-- Two dashes start a one-line comment.

--[[
     Adding two ['s and ]'s makes it a
     multi-line comment.
--]]

-->=============================================================
-- Then, in interactive mode, you can type

--th
-- dofile("class_code2.lua")
--lua <file name>
-->=============================================================
--nil

print(z) 
z = 10
print(z)

-->=============================================================
--The following words are reserved; we cannot use them as identifiers:
--[[
    and       break     do        else      elseif
    end       false     for       function  if
    in        local     nil       not       or
    repeat    return    then      true      until
    while
--]]
-->=============================================================
--[[Lua provides the following relational operators:
    <   >   <=  >=  ==  ~=
--]]
-->=============================================================
--Concatination
H = "Hello"
print(H .. " World")   --> Hello World
print(H)               --> Hello

-->=============================================================
-- Operator 
--[[
  ^
  not  
  *   /
  +   -
  ..
  <   >   <=  >=  ~=  ==
  and
  or

--]]

-->=============================================================
-- Swap, Replace and assign

e, e1 = e1, e             
   
a1, b1, c1 = 0, 1
print(a1,b1,c1)          
aa, bb = 2*a1, 2*b1, 2*b1+1   
print(aa,bb)             
a2, b2, c2 = 0
print(a2,b2,c2)     
a3, b3, c3 = 0, 0, 0
print(a3,b3,c3)  