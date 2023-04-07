import math 
import numpy as np 

def semiTensorProduct(a, b): 
    p = np.shape(a)[1]
    q = np.shape(b)[0]
    v = math.lcm(p, q) 

    m_left = np.kron(a, np.identity(v//p))
    m_right = np.kron(b, np.identity(v//q)) 

    return  np.matmul(m_left, m_right)  


m_not = np.matrix([[0, 1],[1, 0]])
m_or = np.matrix([[1, 1, 1, 0],[0, 0, 0, 1]]) 
m_and = np.matrix([[1, 0, 0, 0],[0, 1, 1, 1]]) 
i_2 = np.identity(2)
i_4 = np.identity(4)  
m_reduce_power = np.matrix([[1, 0], [0, 0], [0, 0], [0, 1]])
w_22 = np.matrix([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]])           

print(m_not)
print(m_or)
print(m_and)
print(i_2)
print(m_reduce_power)   
print(w_22)  

mf = semiTensorProduct(m_or, m_and)
mf = semiTensorProduct(mf, (np.kron(i_2, m_not)))
mf = semiTensorProduct(mf, (np.kron(i_4, m_and)))  
mf = semiTensorProduct(mf, (np.kron(i_2, w_22)))    
mf = semiTensorProduct(mf, (np.kron(i_4, m_reduce_power)))



print(mf)