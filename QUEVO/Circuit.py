
import qutip
from qiskit.quantum_info import partial_trace
from qiskit.visualization import circuit_drawer
from scipy.special import comb
import itertools
import typing
import qiskit.quantum_info as qi
import numpy as np
from typing import List
from qiskit import QuantumCircuit, Aer, assemble, execute
from scipy.special import rel_entr
from .Chromosome import Chromosome
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import StatevectorSimulator




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

        n_qubits = 3
        self.chromosome = chromosome
        self._circuit = QuantumCircuit(n_qubits, 1) #three qubits and 1 classical bit
        self._SHOTS = 2096

  
        #self._STARTING_STATES = np.random.rand(n_states, 2**n_qubits)
        self._STARTING_STATES = [[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 0],
                                 [1, 0, 1],
                                 [1, 1, 0],
                                 [1, 1, 1]]
        self.results = {}

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
        n_qubits = 3
        gates = int(self.chromosome.get_length() / n_qubits)
        gate_dict = self.chromosome.get_gate_dict()

        for i in range(0, gates):
            gate_index = i * n_qubits

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



    def find_chromosome_fitness(self, target_entanglement) -> float:
        backend = Aer.get_backend('statevector_simulator')
        job = execute(self._circuit, backend)
        result = job.result()
        statevector = result.get_statevector()

        entanglement = self.compute_MW_entanglement(statevector)
        fitness = abs(entanglement - target_entanglement)

        return fitness


    def compute_MW_entanglement(self, statevector: np.ndarray) -> float:
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
        n_qubits = 3
        ket = np.reshape(statevector, [2] * n_qubits)  # Reshape the statevector to a tensor
        entanglement_sum = 1
        for k in range(n_qubits):
            rho_k_sq = np.abs(np.trace(np.transpose(ket, axes=np.roll(range(n_qubits), -k))))
            entanglement_sum += rho_k_sq

        entanglement = 1 * (1 - (1 / n_qubits) * entanglement_sum)
        return entanglement

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
