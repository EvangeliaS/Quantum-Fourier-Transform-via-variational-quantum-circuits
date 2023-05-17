from qutip_qip.vqa import VQA

VQA_circuit = VQA(
            num_qubits=1,
            num_layers=1,
            cost_method="OBSERVABLE",
        )
# testcode::

from qutip_qip.vqa import VQA, VQABlock
from qutip import tensor, sigmax

VQA_circuit = VQA(num_qubits=1, num_layers=1)

R_x_block = VQABlock(
    sigmax() / 2, name="R_x(\\theta)"
)

VQA_circuit.add_block(R_x_block)


# .. plot::
#   :context:

from qutip_qip.vqa import VQA, VQABlock
from qutip import sigmax, sigmaz
circ = VQA(num_qubits=1, cost_method="OBSERVABLE")

  #Picking the Pauli Z operator as our cost observable, our circuit's cost function will be: :math:`\langle\psi(t)| \sigma_z | \psi(t)\rangle`

circ.cost_observable = sigmaz()


  #Adding a Pauli X operator as a block to the circuit, the operation of the entire circuit becomes: :math:`e^{-i t X /2}`.


circ.add_block(VQABlock(sigmax() / 2))

  #We can now try to find a minimum in our cost function using the SciPy in-built L-BFGS-B (L-BFGS with box constraints) method. We specify the bounds so that our parameter is :math:`0 \leq t \leq 4`.

result = circ.optimize_parameters(method="L-BFGS-B", use_jac=True, bounds=[[0, 4]])

  #Accessing ``result.res.x``, we have the array of parameters found during optimization. In our case, we only had one free parameter, so we examine the first element of this array.

angle = round(result.res.x[0], 2)
print(f"Angle found: {angle}")
#Angle found: 3.14

  #Finally, we can plot our the measurement outcome probabilities of our circuit after optimization.

result.plot()

