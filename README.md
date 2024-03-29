# Quantum Fourier Transform via variational quantum circuits

This repository contains the code and documentation for my thesis work on using variational quantum circuits (VQC) to simulate the Quantum Fourier Transform (QFT). The aim of this research is to develop a VQC-based approach to efficiently perform QFT, a crucial operation in quantum computing.

## Table of Contents

- [Introduction](#introduction)
- [Key Findings](#key-findings)
- [Getting Started](#getting-started)
- [Ansatz and Circuit Structure](#ansatz-and-circuit-structure)
- [Cost Function](#cost-function)
- [Results](#results)

## Introduction

In this [thesis](https://github.com/EvangeliaS/Quantum-Fourier-Transform-via-variational-quantum-circuits/blob/3e2a6da4b577a2c1c4828fc98ed88ab024080860/Implementing_Quantum_Fourier_Transform_via_variational_quantum_circuits.pdf) , I explore the use of variational quantum circuits (VQC) with adjustable parameters to simulate the Quantum Fourier Transform (QFT). The primary objectives of this research are to reduce the depth of the quantum circuit required for QFT simulation and optimize the circuit parameters using classical algorithms.

## Key Findings

- Increasing the number of trainable parameters generally improves the optimization process, resulting in lower cost values.
- Initial parameter values have a significant impact on convergence rates and final cost values.
- Epsilon values used in optimization algorithms affect execution time and convergence, depending on initialization values.
- Efficient training of the quantum circuit with 26 parameters is possible, and adding more parameters may not significantly improve results.
- An ansatz based on algebraic properties of QFT reduced the number of parameters and circuit depth effectively.


## Ansatz and Circuit Structure

The VQC ansatz is built based on the algebraic properties of the QFT. We decompose the QFT into a set of 11 generators and construct a parametrized circuit using single-qubit and two-qubit gates. This parametrized circuit aims to minimize the cost function by adjusting its parameters.

For detailed information on the ansatz and circuit structure, refer to [Chapter: Building an Ansatz] in the [thesis](https://github.com/EvangeliaS/Quantum-Fourier-Transform-via-variational-quantum-circuits/blob/3e2a6da4b577a2c1c4828fc98ed88ab024080860/Implementing_Quantum_Fourier_Transform_via_variational_quantum_circuits.pdf).

## Cost Function

The cost function measures the distance between the parametrized circuit and the target QFT operation. It is designed to be minimized during the optimization process. More information about the cost function can be found in [Chapter: Cost Function] in the thesis.

## Results

The results of the experiments are presented in [Chapter: Results] in the thesis. These results showcase the impact of various factors such as the number of parameters, initial values, and epsilon values on the optimization process and final cost values.
