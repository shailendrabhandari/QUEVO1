
import qutip
from qiskit.quantum_info import partial_trace
from qiskit.visualization import circuit_drawer
from scipy.special import comb
import itertools
import typing
import numpy as np
from typing import List
from qiskit import QuantumCircuit, Aer, assemble
from scipy.special import rel_entr
from .Chromosome import Chromosome


class Circuit(object):
    """
    A qiskit QuantumCircuit made from a chromosome.

    Attributes
    ----------
    chromosome (Chromosome):
        The integer string representation of the _circuit.
    _circuit (Qiskit.QuantumCircuit):
        Qiskit representation of the chromosome. A _circuit that can be run and simulated.
    _SHOTS (int):
        Number of runs in the qiskit quantum _circuit simulator.
    _STARTING_STATES (List[list]):
        a list of all possible starting states for the 3 qubits.
    """

    def __init__(self, chromosome: Chromosome):
        """
        Circuit constructor. Takes a chromosome as parameter, and creates a Qiskit
        QuantumCircuit object form it.

        Parameters
        ----------
        chromosome: (Chromosome)
            The chromosome that describes the QuantumCircuit.
        """
        self.chromosome = chromosome
        self._circuit = QuantumCircuit(3, 1) #three qubits and 1 classical bit
        self._SHOTS = 2048
        self._STARTING_STATES = [[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 0],
                                 [1, 0, 1],
                                 [1, 1, 0],
                                 [1, 1, 1]]
        self.results = {}


    def __repr__(self):
        """Returns a string visualizing the quantum _circuit"""
        return self.draw()

    def generate_circuit(self) -> None:

        
        """
        Parses the chromosome, and generates a Qiskit QuantumCircuit from it.

	The generate_circuit method generates a Qiskit quantum circuit from a chromosome object.
	The chromosome object has a list of integers and a dictionary that maps the integers to different
	quantum gates, such as 'h', 'cx', 'x', 'swap', 'rzz', 'rxx', 'toffoli', 'y', and 'z'. The method
	 creates a circuit by iterating through the integer list in steps of 3 and using each group of 3
	 integers to determine which gate to apply, to which qubits, and with what parameters. The method then
	 adds the chosen gate to the circuit. Finally, it adds a measurement gate to the first
	 qubit and outputs qubit 0.
        """
        gates = int(self.chromosome.get_length() / 3)

        gate_dict = self.chromosome.get_gate_dict()

        for i in range(0, gates):
            gate_index = i * 3

            a = self.chromosome.get_integer_list()[gate_index]
            b = self.chromosome.get_integer_list()[gate_index + 1]
            c = self.chromosome.get_integer_list()[gate_index + 2]

            gate = gate_dict[str(a)]

            if gate == 'h':
                self._circuit.h(b)
            elif gate == 'cx':
                self._circuit.cx(b, c)
            elif gate == 'x':
                self._circuit.x(b)
            elif gate == 'swap':
                self._circuit.swap(b, c)
            elif gate == 'rzz':
                theta = self.chromosome.get_theta_list()[i]
                self._circuit.rzz(theta=theta, qubit1=b, qubit2=c)
            elif gate == 'rxx':
                theta = self.chromosome.get_theta_list()[i]
                self._circuit.rxx(theta=theta, qubit1=b, qubit2=c)
            elif gate == 'toffoli':
                target = b
                if target == 1:
                    self._circuit.toffoli(0, 2, target)
                else:
                    self._circuit.toffoli(abs(target - 1), abs(target - 2), target)
            elif gate == 'y':
                self._circuit.y(b)
            elif gate == 'z':
                self._circuit.z(b)
            else:
                print(gate + " is not a valid gate!")

        self._circuit.measure(0, 0) 


    def compute_MW_entanglement(self, ket):
        """
        Compute the Mayer-Wallach measure of entanglement.
        
        Parameters:
        -----------
        ket : numpy.ndarray or list
            Vector of amplitudes in 2**N dimensions
        
        Returns:
        --------
        MW_entanglement : float
            Mayer-Wallach entanglement value for the input ket
        """
        ket = qutip.Qobj(ket, dims=[[2]*(3), [2]*(3)]).unit()
        entanglement_sum = 2
        for k in range(3):
            rho_k_sq = ket.ptrace([k])**2
            entanglement_sum += rho_k_sq.tr()
   
        MW_entanglement = 2*(1 - (1/3)*entanglement_sum)
        return MW_entanglement

    def find_chromosome_fitness(self, state: List[int]) -> float:
        """
        Compute the fitness of a chromosome by computing the average Mayer-Wallach
        entanglement for all the starting states.
        
        Parameters:
        -----------
        state : list
            A binary string of length 3 representing the state of the qubits
            
        Returns:
        --------
        avg_MW_entanglement : float
            The average Mayer-Wallach entanglement for all starting states
        """

        self.generate_circuit()

        MW_entanglement_values = []
        fitness = 0
        MW_entanglement = self.compute_MW_entanglement(ket=state)
        print('MW_entanglement = {}\n'.format(MW_entanglement))
        fitness = 0 + MW_entanglement
        return fitness
 
    '''@staticmethod
    def scott_helper(state, perms):
        """Helper function for entanglement measure. It gives trace of the output state"""
        dems = np.linalg.matrix_power(
            [partial_trace(state, list(qb)).data for qb in perms], 2
        )
        trace = np.trace(dems, axis1=1, axis2=2)
        return np.sum(trace).real

    def find_chromosome_fitness(self, states):
        r"""Returns the meyer-wallach entanglement measure for the given circuit.

        .. math::
            Q = \frac{2}{|\vec{\theta}|}\sum_{\theta_{i}\in \vec{\theta}}
            \Bigg(1-\frac{1}{n}\sum_{k=1}^{n}Tr(\rho_{k}^{2}(\theta_{i}))\Bigg)

        """
        fitness = 0
        permutations = list(itertools.combinations(range(3), 3 - 1))
        ns = 2 * sum(
            [
                1 - 1 / 3 * self.scott_helper(state, permutations)
                for state in states
            ]
        )
        return ns.real
        fitness = 0 + ns.real
        print('MW_entanglement = {}\n'.format(ns))
        return fitness'''

    def run_simulator(self) -> dict:
        """
        Runs the _circuit on the Qiskit AER simulator and returns the results as a dictionary.

        Returns
        -------
        counts: dict
            The results from the AER simulation.
        """
        aer_sim = Aer.get_backend('aer_simulator')
        # aer_sim = Aer.get_backend('aer_simulator_density_matrix')
        # aer_sim = Aer.get_backend('aer_simulator_stabilizer')
        quantum_circuit = assemble(self._circuit, shots=self._SHOTS)
        job = aer_sim.run(quantum_circuit)
        counts = job.result().get_counts()
        return counts

    def draw(self) -> None:
        """Prints a visual representation of the _circuit"""
        print(self._circuit.draw(output='text'))

    def clear_circuit(self) -> None:
        """Clears the Qiskit QuantumCircuit for all gates"""
        self._circuit.data.clear()
