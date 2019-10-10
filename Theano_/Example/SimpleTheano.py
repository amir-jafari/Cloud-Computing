import numpy as np
import theano.tensor as T
from theano import function
from theano import pp
from theano.printing import debugprint as dbp
from theano.printing import pydotprint

a = T.matrix('a')
b = T.matrix('b')
z = (a + b)/2
average = function([a, b], z)


aa1 = np.array([[3]])
bb1 = np.array([[4.5]])


aa2 = np.array([[4,8]])
bb2 = np.array([[2,9]])

aa3 = np.array([[1., 2], [3, 4], [5, 6]])
bb3 = np.array([[3., 6], [5, 2], [2, 7]])

#aa=[[1, 2], [3, 4]]
#bb = [[10, 20], [30, 40]]
#aa = [[1]]
#bb=[[2]]

result1 = average(aa1,bb1)
print(result1)

result2 = average(aa2,bb2)
print(result2)





print(pp(z))

result3 = average(aa3,bb3)
print(result3)

print(dbp(z))
print(result3)

#pydotprint(average, outfile="./average_graph.png", var_with_name_simple=True)