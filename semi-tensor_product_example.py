import math 
import numpy as np 

def semiTensorProduct(a, b): 
    n = np.shape(a)[1]
    p = np.shape(b)[0]
    v = math.lcm(n, p) 

    m_left = np.kron(a, np.identity(v//n))
    m_right = np.kron(b, np.identity(v//p)) 

    return  np.matmul(m_left, m_right)  

#calculate semi tensor product for Boolean expressions  
def getBooleanBySemiTensorProduct(mf, X): 
    value = mf 
    
    numCols = np.shape(mf)[1]   
    numRegs = np.shape(X)[1]
    #calculate semi tensor product for each variable in b 
    for i in range(numRegs): 
        numCols = numCols//2 
        b0 = X[0,i] 
        b1 = X[1,i] 
        value = value[:,:numCols]*b0 + value[:,numCols:]*b1         
    return value   


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
 
x1 = np.matrix([[1], [0]])
x2 = np.matrix([[1], [0]])
x3 = np.matrix([[1], [0]])    

X = np.hstack((x1,x2,x3))  
print(X)   
print(x1)      

a = semiTensorProduct(mf, x1)
print(a) 
b = semiTensorProduct(a, x2) 
print(b)
c = semiTensorProduct(b, x3)   
print(c) 


#print(semiTensorProduct(mf, X))  

print(getBooleanBySemiTensorProduct(mf, X)) 
