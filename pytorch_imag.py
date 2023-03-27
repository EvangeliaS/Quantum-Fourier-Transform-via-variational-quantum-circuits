import torch
import numpy as np

ss1 = [[0,1],[1,0]]
ss1 = torch.tensor(ss1)
#print(ss1)

#complex matrix with real and imaginary parts
ss2 = [[0,-1j],[1j,0]]
ss2 = torch.tensor(ss2)
#print(ss2)

ss3 = [[1,0],[0,-1]]
ss3 = torch.tensor(ss3)
#print(ss3)

ss4 = torch.eye(2)
#print(ss4)

#fill a tensor 1*8 with 1/2*sqrt(2)
QF3 = torch.full((1,8),1/2*torch.sqrt(torch.tensor(2.0)))
#QF3 = torch.tensor(QF3)
#print(QF3)

#fill a tensor 1*8 with 1/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), - 1/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2), -(1j)/2*sqrt(2), exponential(-1j*pi/4)/2*sqrt(2)
QF4 = [[1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)), 1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)),-1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)) , -1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0))]]
QF4 = torch.tensor(QF4)
#print(QF4)

#fill a tensor 1*8, with 1/2*sqrt(2), (1j)/2*sqrt(2), - 1/2*sqrt(2),  -(1j)/2*sqrt(2), 1/2*sqrt(2), (1j)/2*sqrt(2), - 1/2*sqrt(2),  -(1j)/2*sqrt(2)
QF5 = [[1/2*torch.sqrt(torch.tensor(2.0)), 1j/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), -1j/2*torch.sqrt(torch.tensor(2.0)), 1/2*torch.sqrt(torch.tensor(2.0)), 1j/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), -1j/2*torch.sqrt(torch.tensor(2.0))]]
QF5 = torch.tensor(QF5)
#print(QF5)

#fill a tensor 1*8, with 1/2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), -(1j)/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2), - 1/2*sqrt(2),  exponential(-1j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2)
QF6 = [[1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)), -1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)),-1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)) , 1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0))]]
QF6 = torch.tensor(QF6)
#print(QF6)

#fill a tensor 1*8 with 1/2*sqr(2) and -1/2*sqr(2)
QF7 = [[1/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), 1/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), 1/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), 1/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0))]]
QF7 = torch.tensor(QF7)
#print(QF7)

QF8 = [[1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)), 1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)) , -1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0))]]
QF8 = torch.tensor(QF8)
#print(QF8)

#fill a tensor 1*8, with 1/2*sqrt(2), (-1j)/2*sqrt(2), - 1/2*sqrt(2),  (1j)/2*sqrt(2), 1/2*sqrt(2), (-1j)/2*sqrt(2), - 1/2*sqrt(2),  (1j)/2*sqrt(2)
QF9 = [[1/2*torch.sqrt(torch.tensor(2.0)), -1j/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), 1j/2*torch.sqrt(torch.tensor(2.0)), 1/2*torch.sqrt(torch.tensor(2.0)), -1j/2*torch.sqrt(torch.tensor(2.0)), -1/2*torch.sqrt(torch.tensor(2.0)), 1j/2*torch.sqrt(torch.tensor(2.0))]]
QF9 = torch.tensor(QF9)
#print(QF9)


#fill a tensor 1*8 with 1/2*sqrt(2), exponential(-1j*pi/4)/2*sqrt(2), (-1j)/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2), - 1/2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2)
QF10 = [[1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)), -1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(-3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)),-1/2*torch.sqrt(torch.tensor(2.0)), torch.exp(3j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0)) , 1j/2*torch.sqrt(torch.tensor(2.0)), torch.exp(1j*torch.tensor(np.pi/4))/2*torch.sqrt(torch.tensor(2.0))]]
QF10 = torch.tensor(QF10)
#print(QF10)

#make tensor with all the above tensors in it
QF = torch.stack([QF3, QF4, QF5, QF6, QF7, QF8, QF9, QF10], 0)
print(QF)
#find conjugate transpose of QF3
B = torch.conj(torch.transpose(QF, 1,0))

#print(B)

#find kronecker product of QF3 and B
C = torch.kron(QF, B)
#print(C)