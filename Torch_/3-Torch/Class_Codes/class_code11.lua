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
-- Hard Tanh
require 'nn'
require 'gnuplot'

x = torch.linspace(-2, 2)
Hard_tan = nn.HardTanh()
Output = Hard_tan:forward(x)
Ones= torch.ones(100)
derv = Hard_tan:backward(x, Ones)

gnuplot.figure(1)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('Hard Tanh Transfer Function')


-->=============================================================
-- Hard Shrink
x = torch.linspace(-2, 2)
Hard_Shrink = nn.HardShrink()
Output = Hard_Shrink:forward(x)
Ones= torch.ones(100)
derv = Hard_Shrink:backward(x, Ones)

gnuplot.figure(2)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('Hard Shrink Transfer Function')

-->=============================================================
-- Soft Shrink
x = torch.linspace(-2, 2)
Soft_Shrink = nn.SoftShrink()
Output = Soft_Shrink:forward(x)
Ones= torch.ones(100)
derv = Soft_Shrink:backward(x, Ones)

gnuplot.figure(3)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('Soft Shrink Transfer Function')

-->=============================================================
-- ReLU
x = torch.linspace(-2, 2)
ReLU = nn.ReLU()
Output = ReLU:forward(x)
Ones= torch.ones(100)
derv = ReLU:backward(x, Ones)

gnuplot.figure(4)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('ReLU Transfer Function')

-->=============================================================
-- ReLU6
x = torch.linspace(-2, 2)
RELU_M = nn.ReLU6()
Output = RELU_M:forward(x)
Ones= torch.ones(100)
derv = RELU_M:backward(x, Ones)

gnuplot.figure(5)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('ReLU6 Transfer Function')

-->=============================================================
-- RReLU
x = torch.linspace(-2, 2)
RRELU = nn.RReLU()
Output = RRELU:forward(x)
Ones= torch.ones(100)
derv = RRELU:backward(x, Ones)

gnuplot.figure(6)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('RReLU Transfer Function')

-->=============================================================
-- Tanh
x = torch.linspace(-2, 2)
TanH = nn.Tanh()
Output = TanH:forward(x)
Ones= torch.ones(100)
derv = TanH:backward(x, Ones)

gnuplot.figure(7)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('Tanh Transfer Function')

-->=============================================================
-- Sigmoid
x = torch.linspace(-2, 2)
Sigmoid = nn.Sigmoid()
Output = Sigmoid:forward(x)
Ones= torch.ones(100)
derv = Sigmoid:backward(x, Ones)

gnuplot.figure(8)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('Sigmoid Transfer Function')

-->=============================================================
-- LogSoftMax
x = torch.linspace(-2, 2)
LogSoftMax = nn.LogSoftMax()
Output = LogSoftMax:forward(x)
Ones= torch.ones(100)
derv = LogSoftMax:backward(x, Ones)

gnuplot.figure(9)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('LogSoftMax Transfer Function')

-->=============================================================
-- SoftSign
x = torch.linspace(-2, 2)
SoftSign = nn.SoftSign()
Output = SoftSign:forward(x)
Ones= torch.ones(100)
derv = SoftSign:backward(x, Ones)

gnuplot.figure(10)
gnuplot.plot({'f(x)', x, Output, '+-'}, {'fdot(x)', x, derv, '+-'})
gnuplot.grid(true)
gnuplot.title('SoftSign Transfer Function')

