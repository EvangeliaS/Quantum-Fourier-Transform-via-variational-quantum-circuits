Variational quantum circuits are quantum circuits which contain gates with adjustable parameters. 
Such circuits are already physically realizable in small scale and there is a wide
range of possible applications for these promising structures. In this thesis, I develop the
idea of tuning a variational quantum circuit to simulate the important for quantum com-
puting, operation of Quantum Fourier Transform. I use algebraic arguments, so called an
ansatz, for reducing the depth of the variational quantum circuit and use different classical algorithms to optimize the parameters. The results of this thesis concerning 3-qubit
circuits can be possibly extended to a higher number of qubits.

# Variational Quantum Circuits for Quantum Fourier Transform Simulation

This repository contains the code and documentation for my thesis work on using variational quantum circuits (VQC) to simulate the Quantum Fourier Transform (QFT). The aim of this research is to develop a VQC-based approach to efficiently perform QFT, a crucial operation in quantum computing.

## Table of Contents

- [Introduction](#introduction)
- [Key Findings](#key-findings)
- [Getting Started](#getting-started)
- [Ansatz and Circuit Structure](#ansatz-and-circuit-structure)
- [Cost Function](#cost-function)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this thesis, I explore the use of variational quantum circuits (VQC) with adjustable parameters to simulate the Quantum Fourier Transform (QFT). The primary objectives of this research are to reduce the depth of the quantum circuit required for QFT simulation and optimize the circuit parameters using classical algorithms.

## Key Findings

- Increasing the number of trainable parameters generally improves the optimization process, resulting in lower cost values.
- Initial parameter values have a significant impact on convergence rates and final cost values.
- Epsilon values used in optimization algorithms affect execution time and convergence, depending on initialization values.
- Efficient training of the quantum circuit with 26 parameters is possible, and adding more parameters may not significantly improve results.
- An ansatz based on algebraic properties of QFT reduced the number of parameters and circuit depth effectively.


## Ansatz and Circuit Structure

The VQC ansatz is built based on the algebraic properties of the QFT. We decompose the QFT into a set of 11 generators and construct a parametrized circuit using single-qubit and two-qubit gates. This parametrized circuit aims to minimize the cost function by adjusting its parameters.

For detailed information on the ansatz and circuit structure, refer to [Chapter: Building an Ansatz](#building-an-ansatz) in the thesis.

## Cost Function

The cost function measures the distance between the parametrized circuit and the target QFT operation. It is designed to be minimized during the optimization process. More information about the cost function can be found in [Chapter: Cost Function](#cost-function) in the thesis.

## Results

The results of the experiments are presented in [Chapter: Results](#results) in the thesis. These results showcase the impact of various factors such as the number of parameters, initial values, and epsilon values on the optimization process and final cost values.
