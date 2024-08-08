# 🌟 QUEVO1: Evolutionary Computing for Quantum Circuit Generation and Entanglement Measurement 🌟

This code was originally created for a pilot project conducted by the [NordSTAR](https://www.oslomet.no/nordstar) research group at Oslo Metropolitan University. Its main purpose was to generate simulated quantum circuits to achieve a desired probability outcome for eight initial conditions. However, it has been further modified to include the measurement of entanglement in the generated quantum circuits using MW-measure of entanglement methods.

## 📜 Table of Contents
- [General Info](#general-info)
- [Technologies](#technologies)
- [User Guide](#user-guide)
- [Setup](#setup)
- [Publication](#publication)

## ℹ️ General Info
The QUEVO1 framework utilizes an evolutionary algorithm to automatically generate quantum circuits that satisfy defined properties specified through a fitness function. The parameters of evolution can be controlled by adjusting factors such as the initial population, number of generations, probability of mutation operator, number of gates in the circuit, and the chosen fitness function.

This code creates quantum circuits using the Qiskit framework and measures the entangling capability of the circuits using the MW (Meyer-Wallach) entanglement measure. The MW entanglement measure is a measure of entanglement between qubits in a quantum system. The higher the value of the MW measure (which ranges from 0 to 1), the more entangled the system is.

If you wish to experiment with the framework, you can access the Example project notebook at the following link: [Example project notebook](https://github.com/shailendrabhandari/QUEVO1/blob/main/Example%20project.ipynb). The notebook provides instructions on how to use the framework and test its functionality.

### 🛠️ Structure of the Components
The package consists of three main classes: `Chromosome`, `Generation`, and `Circuit`. `Chromosome` and `Generation` represent parts of the genetic algorithm, while the `Circuit` handles generating quantum circuits, simulations, and measurements done with Qiskit.

#### 🧬 Chromosome
This is the core of the algorithm and the integer representation. It contains the list of integers, and all functions for generating and mutating the lists.

#### 🌱 Generation
Stores several chromosomes. It can populate itself with a given number of generated chromosomes. Also has functions to compare the different chromosomes' fitness and stores it in the fitness list.

#### 🔄 Circuit
Handles the parsing from integer list to circuit and initializing Qiskit code to simulate the circuits.

### 💻 Quantum Circuit as a List of Integers
Quantum computing is a rapidly developing field that has great potential to revolutionize the way we process information. It uses quantum mechanics principles to perform computations that are not possible on classical computers.

One way to represent a quantum circuit is to use a list of integers, where each group of three successive integers represents a gate. The first integer in the group denotes the type of gate, the second integer represents the qubit to which the gate is applied, and the third integer represents the qubit that controls the gate.

For example:
- A Pauli-X gate on qubit 0 can be represented by the list `[1, 0, 0]`, where `1` corresponds to the X-gate.
- A CNOT gate with control qubit 0 and target qubit 1 can be represented by the list `[2, 1, 0]`.
- A Hadamard gate on qubit 2 can be represented by the list `[3, 2, 0]`.

The correspondence between integers and gates is determined dynamically based on the gates provided in the code, and a table is automatically generated each time the code is run.

It's worth noting that not all gates have a control qubit, and in those cases, the third integer in the group is ignored.

Using a list of integers to represent a quantum circuit can be useful for various purposes, such as circuit visualization, circuit optimization, and circuit compilation. However, it's important to keep in mind that this is just one of many possible representations of a quantum circuit, and different representations may be more suitable for different applications.

### 🔢 Example

A list of integers representing three qubits and their position. Here we assume that the gates used are: Hadamard (`h`) represented by `0`, Pauli X (`cx`) by `1`, and CNOT (`cx`) by `2`.

The list in this example is: `[1, 0, 1, 0, 2, 1, 2, 1, 2]`. That means Gate 1 = 101, Gate 2 = 021, and Gate 3 = 212.

- Gate 1's first integer is `1`, which means Pauli-X gate. The second integer tells us it is placed on the `0th` qubit, and since the X gate is a single qubit gate, we ignore the last number.

    ![One gate](https://github.com/Overskott/Evolving-quantum-circuits/blob/main/Images/X-gate.png)

- The second triplet of integers tells us that the second gate should be a Hadamard and be placed on the `2nd` qubit. Again, we ignore the last integer.

    ![Two gates](https://github.com/Overskott/Evolving-quantum-circuits/blob/main/Images/H-gate.png)

- The last triplet shows us that the quantum circuit has a CNOT gate with target on qubit `2`. Here the last integer is included and tells us that the control is qubit `2`. We now have the complete string parsed to a quantum circuit:

    ![Three gates](https://github.com/Overskott/Evolving-quantum-circuits/blob/main/Images/CX-gate.png)

For the moment the supported gates are: Hadamard, Pauli gates (X, Y and Z), Cnot, Toffoli, Swap, RZZ, and RXX.
 
Some gates (RZZ, RXX) also need an angle value (theta (0, 2pi)), which are automatically generated, updated, and stored in a separate list from the integers. 

## 👥 User Guide

I recommend taking a look at the notebook [Example project](https://github.com/shailendrabhandari/QUEVO1/blob/main/Example%20project.ipynb).

If you want to learn more in detail about all the functions, please read the docstrings.

## 🛠️ Technologies
Project is created with:
- **Python version:** 3.8 
- **Qiskit version:** qiskit-0.41.0 qiskit-aer-0.11.2 qiskit-ibmq-provider-0.20.0 qiskit-terra-0.23.1 rustworkx-0.12.1 stevedore-5.0.0 symengine-0.9.2
- **scipy.special:** 1.7.1

## ⚙️ Setup
This project uses Qiskit and scipy. The best way of installing Qiskit is by using pip:
```bash
$ pip install qiskit
$ pip install scipy
```

## 📚 Publication

If you find our work useful, please consider citing our publication:

```bibtex
@misc{bhandari2024quantumevolutionaryalgorithmshelp,
      title={How quantum and evolutionary algorithms can help each other: two examples}, 
      author={Shailendra Bhandari and Stefano Nichele and Sergiy Denysov and Pedro G. Lind},
      year={2024},
      eprint={2408.00448},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2408.00448}, 
}
```


