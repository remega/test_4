import numpy as np
import math

# a = [[1,2,3],[4,5,6]]
# b = a[0][1]
# c = a[0,1]
a = [1,"nan",2,"nan"]
for i in a:
    if math.isnan(float(i)) is False:
        print(i)

# print(a,b)
