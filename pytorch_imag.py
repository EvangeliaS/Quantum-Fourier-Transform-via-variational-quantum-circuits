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

#fill a tensor 1*8 with 1/(2*sqrt(2)
QF3 = torch.full((1,8),1/(2*torch.sqrt(torch.tensor(2.0))))
#QF3 = torch.tensor(QF3)
#print(QF3)

#fill a tensor 1*8 with 1/(2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), - 1/(2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2), -(1j)/2*sqrt(2), exponential(-1j*pi/4)/2*sqrt(2)
QF4 = [[1/(2*torch.sqrt(torch.tensor(2.0))), (torch.exp(1j*torch.tensor(np.pi/4)))/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))) , torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))),-1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF4 = torch.tensor(QF4)
#print(QF4)

#fill a tensor 1*8, with 1/(2*sqrt(2), (1j)/2*sqrt(2), - 1/(2*sqrt(2),  -(1j)/2*sqrt(2), 1/(2*sqrt(2), (1j)/2*sqrt(2), - 1/(2*sqrt(2),  -(1j)/2*sqrt(2)
QF5 = [[1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0)))]]
QF5 = torch.tensor(QF5)
#print(QF5)

#fill a tensor 1*8, with 1/(2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), -(1j)/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2), - 1/(2*sqrt(2),  exponential(-1j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2)
QF6 = [[1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))),-1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , 1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF6 = torch.tensor(QF6)
#print(QF6)

#fill a tensor 1*8 with 1/(2*sqr(2) and -1/(2*sqr(2)
QF7 = [[1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0)))]]
QF7 = torch.tensor(QF7)
#print(QF7)

QF8 = [[1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF8 = torch.tensor(QF8)
#print(QF8)

#fill a tensor 1*8, with 1/(2*sqrt(2), (-1j)/2*sqrt(2), - 1/(2*sqrt(2),  (1j)/2*sqrt(2), 1/(2*sqrt(2), (-1j)/2*sqrt(2), - 1/(2*sqrt(2),  (1j)/2*sqrt(2)
QF9 = [[1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0)))]]
QF9 = torch.tensor(QF9)
#print(QF9)


#fill a tensor 1*8 with 1/(2*sqrt(2), exponential(-1j*pi/4)/2*sqrt(2), (-1j)/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2), - 1/(2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2)
QF10 = [[1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))),-1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , 1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF10 = torch.tensor(QF10)
#print(QF10)

#make tensor with all the above tensors in it
QF = torch.cat((QF3, QF4, QF5, QF6, QF7, QF8, QF9, QF10), 0)
print("QF")
print(QF)
#find conjugate transpose of QF3
B = torch.conj(torch.transpose(QF, 0,1))

#print(B)

#find kronecker products
c1 = torch.kron(ss1, ss4)
c1 = torch.kron(c1, ss4)

c2 = torch.kron(ss4, ss2)
c2 = torch.kron(c2, ss4)

c3 = torch.kron(ss4, ss3)
c3 = torch.kron(c3, ss4)

c4 = torch.kron(ss4, ss1)
c4 = torch.kron(c4, ss4)

c5 = torch.kron(ss4, ss4)
c5 = torch.kron(c5, ss3)

c6 = torch.kron(ss4, ss3)
c6 = torch.kron(c6, ss3)

c7 = torch.kron(ss4, ss1)
c7 = torch.kron(c7, ss3)

c8 = torch.kron(ss1, ss1)
c8 = torch.kron(c8, ss4)

c9 = torch.kron(ss1, ss2)
c9 = torch.kron(c9, ss4)

c10 = torch.kron(ss3, ss4)
c10 = torch.kron(c10, ss2)

c11 = torch.kron(ss1, ss4)
c11 = torch.kron(c11, ss3)

#print(c1.size())

vv3 = torch.stack((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))

# print("vv3\n")
# print(vv3)

# print(vv3.size())

#set G matrix composed of a matrix for each of the vv3 matrices, multiplied by the i complex number , to the exponent power
G = torch.zeros(11, 8, 8, dtype=torch.complex64)

for i in range(G.size(dim = 0)):
    for j in range(G.size(dim = 1)):
        for k in range(G.size(dim = 2)):
            G[i][j][k] = torch.exp(1j*vv3[i][j][k])

print(G[0])

#check if G is equal to its transpose
g_transpose = torch.transpose(G[0], 0,1)
if torch.equal(G[0], g_transpose):
    print("yes")

#find conjugate transpose of G
g_conj_transpose = torch.conj(torch.transpose(G[0], 0,1))
print(g_conj_transpose)

#find unitary matrix

unit = g_conj_transpose@G[0]
print(unit)

# unit = G[0]@g_conj_transpose
# print(unit)

# #compute the inverse of G
# G_inverse = torch.linalg.inv(G[0])
# print(G_inverse)

# new = G_inverse@G[0]
# print(new)


# print(G[0])
# print(g_conj_transpose)




# def Gx(x):
#or maybe e^x*G
#     return x*G

# x = torch.rand(1, dtype=torch.float32)
# print("X is: \n\n", x)
# GX = Gx(x)
# print("G is: \n\n", G[0])
# print(GX[0])
# #conjugate transpose of Gx(x)
# g_conj_transpose = torch.conj(torch.transpose(GX[0], 0,1))
# print(g_conj_transpose)
# unit = torch.matmul(g_conj_transpose, GX[0])