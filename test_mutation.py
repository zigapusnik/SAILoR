import numpy as np

def test_two_p():
    n = 16
    p1 = 0.1 
    x = 0.094 
    p2 = x*p1/(1 - x) 
    
    AdjM = (np.random.rand(n, n) < np.ones((n, n))*x).astype(int)     
    a = 0    

    print(p1)
    print(p2) 

    for i in range(10):
        print("Number of ones is " + str((AdjM > 0).sum()))  

        rndVals = np.random.rand(n, n)  

        indOnes = np.where(AdjM == 1) 
        indZeros = np.where(AdjM == 0) 

        AdjM[indOnes] = (rndVals[indOnes] > p1).astype(int)       
        AdjM[indZeros] = (rndVals[indZeros] < p2).astype(int)   

def test_one_p():
    n = 64
    p1 = 0.9 
    x = 0.94 

    a = x

    AdjM = (np.random.rand(n, n) < np.ones((n, n))*x).astype(int)     

    print(p1)

    for i in range(1000):
        print("Number of ones is " + str((AdjM > 0).sum()))  


        rndVals = np.random.rand(n, n)  

        indices = np.where(rndVals < p1) 
        AdjM[indices] = AdjM[indices] - 1


        AdjM = np.abs(AdjM)     

if __name__ == "__main__":  
    test_one_p()  
