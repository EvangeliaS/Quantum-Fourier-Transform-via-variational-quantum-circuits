#create a loop that generates circuits with different x values and combination of functions.Gx(x) gates 
#and test the results of the circuits

#generate random x values between 0 and 2pi

# x = torch.rand(10, dtype=torch.float32)
# #generate random circuits: matrix product of functions.Gx(x)(combination of i < 5 and k >=5)
# #and test the results of the circuits
# for i in range(x.size(dim = 0)):
#     functions.Gx_i = functions.Gx((x[i].item()))
#     for j in range(5, functions.Gx_i.size(dim = 0)):
#             for k in range(functions.Gx_i.size(dim = 0)):
#                 if k < 5:

#not single qubit gates after another

#starting with 2-qubit gates(i>5 in G matrix)  + 
# 2 single qubit gates(i<5 in G matrix) afte every 2-qubit gate


#να μην εχουν το 4 στην ιδια θεση(2qubits)

# for vv3 in range(functions.Gx_i.size(dim = 0)):
#     for 2quit_gate in range(functions.Gx_i)


#blocks per 3 and permute them to get the final G matrix
