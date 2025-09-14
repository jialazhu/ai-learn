import numpy as np
import pandas as pd

arr = np.array([1,2,3,4,5])
print(arr)

print(np.arange(1, 6,3))

print(np.zeros((3,3)))

arr1 = np.array([1.0,2.0,3.0,4.0,5.0])
print(f'{arr1 > 2}')


dicts = {
    'name':['zhangsan','xiao','ere'],
    'age':[18,21,33],
    'city':['beijing','shanghai','hangzhou']
}

dic = pd.DataFrame(dicts)
print(dic)

