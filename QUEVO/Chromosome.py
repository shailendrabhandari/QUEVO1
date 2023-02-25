

import copy
import math
import random
from typing import List


class Chromosome(object):
    """
    A class used to represent a quantum computer _circuit as a list of integers.

    The _circuit is represented as a list
    of integers, where each gate is three successive integers i.e.:

    Three gates as list:
    [2, 0, 1, 3, 1 ,1, 4, 0, 2]

    Gate number:               1  |  2  |  3
    Gate int representation:  201 | 311 | 402

    The first int is what kind of gate it is (i.e. X-gate, C-NOT, Hadamard).
    The second int is what qubit is this assigned to (0 -> qubit 0, 1 -> qubit 1, ...)
    The third int is what qubit is controlling the gate. in cases where gates do not have an
    external controller, this int is ignored.


    The table under shows what gates are acceptable and how to notate the in the _gate_list

    | Supported gate types| Notation |
    |---------------------|----------|
    |       Hadamard      |    'h'   |
    |       Pauli X       |    'x'   |
    |       Pauli Z       |    'z'   |
    |       Pauli Y       |    'y'   |
    |        C-NOT        |   'cx'   |
    |      Swap gate      |  'swap'  |
    |       Toffoli       | 'toffoli'|
    |         RXX         |   'rxx'  |
    |         RZZ         |   'rzz'  |

    Some gates (RZZ, RXX) also need an angle value (theta) stored in a separate list.

    Attributes
    ----------
    _integer_list: List[int]
        List of integers representing the quantum _circuit.
    _theta_list: List[float]
        A list of angle values for the gates. This list is the same length as number of gates (len(_integer_list) / 3).
    _length: int
        The number of integers in the integer representation
    _gate_types: List[str]
        A list of all the gates the chromosome is allowed to operate with.
    _gate_dict: dict
        The table that holds gates and integers
    """

    def __init__(self, gate_types: List[str]) -> None:
        """The Chromosome constructor"""
        self._integer_list: List[int] = []
        self._theta_list: List[float] = []
        self._fitness_score: float = 0
        self._length: int = 0
        self._gate_types = gate_types
        self._gate_dict: dict = self._create_gate_dict()
        self._fitness_score = None

    def set_fitness_score(self, fitness_score):
        self._fitness_score = fitness_score + np.random.uniform(0, 0.0001)
    def __repr__(self) -> str:
        """Returns desired for printing == print(_integer_list)"""
        return str(self._integer_list)

    def __len__(self) -> int:
        """Returns the number of int in _integer_list"""
        return self._length

    def __iter__(self) -> List[int]:
        """Returns the iterable _integer_list"""
        yield from self._integer_list


    def __lt__(self, other):
        return self._fitness_score < other.get_fitness_score()

    def _create_gate_dict(self) -> dict:
        """Creates and return a dict of the _gate_types"""
        gate_dict: dict = {}
        for j in range(0, len(self._gate_types)):
            gate_dict[str(j)] = self._gate_types[j]

        return gate_dict

    def set_integer_list(self, integer_list: List[int]) -> None:
        """
        Changes the chromosome's integer list to the one given as parameter.

        Parameters:
           integer_list (List[int]): Quantum _circuit integer representation as list.
        """
        old_integer_list = copy.copy(self._integer_list)
        self.clear()
        for integer in integer_list:
            self._integer_list.append(integer)
        self._update_length()
        if not old_integer_list:
            self._generate_theta_list()
        else:
            self._update_theta_list(old_integer_list, self._integer_list)

    def get_gate_dict(self) -> dict:
        """Returns the chromosome's _gate_dict attribute"""
        return self._gate_dict

    def get_integer_list(self) -> List[int]:
        """Returns the list of integers representing the _circuit"""
        return self._integer_list

    def _update_length(self) -> None:
        """Sets _length to the number of integers in _integer_list"""
        self._length = len(self._integer_list)

    def get_length(self) -> int:
        """Returns the number of integers in _integer_list."""
        return self._length

    def get_theta_list(self) -> List[float]:
        """Returns the list of angles in the _circuit"""
        return self._theta_list

    def set_fitness_score(self, score: float) -> None:
        self._fitness_score = score

    def get_fitness_score(self) -> float:
        return self._fitness_score

    def _generate_theta_list(self) -> None:
        """Generates a list of angles based on the current list of integers"""
        self._theta_list.clear()
        gates = int(self._length / 3)

        for i in range(0, gates):
            int_index = i * 3
            gate = self._gate_dict[str(self._integer_list[int_index])]
            if gate in ['rzz', 'rxx']:
                theta = random.uniform(0, 2 * math.pi)
                self._theta_list.append(theta)
            else:
                self._theta_list.append(0)

    def _update_theta_list(self, old_list: List[int], new_list: List[int]) -> None:
        """
        Updates the list of theta values. Used when a _integer_list is changed.
        Takes the _integer.list before the change as old_list, and the _integer_list
        after the change as new_list.

        Parameters
        ----------
        old_list : List[int]
            The list before change to list happened.
        new_list : List[int]
            The new_list list with one or more changed integers.
        """
        gates = int(self._length / 3)
        change_list = self._change_in_theta(old_list, new_list)

        for i in range(0, gates):
            int_index = i * 3
            gate = self._gate_dict[str(self._integer_list[int_index])]
            if change_list[i] == 1 and gate in ['rzz', 'rxx']:
                theta = random.uniform(0, 2 * math.pi)
                self._theta_list[i] = theta
            elif gate in ['rzz', 'rxx']:
                continue
            else:
                self._theta_list[i] = 0

    def _change_in_theta(self, old_list, new_list) -> List[int]:
        """
        Compares the old_list to the new_list and returns a binary string where
        1 indicates a change in the theta value for that gate, 0 indicates no
        change in theta.

        Parameters
        ----------
        old_list : List[int]
            The list before change to list happened.
        new_list : List[int]
            The new_list list with one or more changed integers.

        Returns
        -------
        List[int]
            a list of 1 or 0.
        """

        binary_list = []
        for i in range(0, self._length):
            if old_list[i] == new_list[i]:
                binary_list.append(0)
            else:
                binary_list.append(1)
        return binary_list

    def clear(self) -> None:
        """Clears chromosome's lists and resets _length"""
        self._integer_list.clear()
        self._theta_list.clear()
        self._length = 0

    def generate_random_chromosome(self, gates: int) -> None:
        """
        Generates a random list of integers representing a quantum _circuit with
        the parameter "gates" number of gates

        Parameters
        ----------
        gates : int
            The number of gates in the generated _circuit representation.
        """

        self.clear()
        for i in range(gates * 3):
            if i % 3 == 0:
                self._integer_list.append(random.randrange(0, len(self._gate_types)))
            else:
                self._integer_list.append(random.randrange(0, 3))

        self._update_length()
        self._fix_duplicate_qubit_assignment()
        self._generate_theta_list()

    def mutate_chromosome(self, probability: float) -> 'Chromosome':
        """
        Mutates the chromosome. Mutation can be of either replacing a gates from the pool of gates in the chromosome
        with a randomly generated new one, or replacing the chromosome so as to generate the best four best parents.
        Type of mutation is selected by probability.

        Parameters
        ----------
        probability : (float) optional
            Value between 0 and 1. The probability of replacing a gate.
            The probability of replacing the whole chromosome is 1-probability.

        Returns
        -------
        mutated_chromosome : (Chromosome)
            The mutated chromosome.
        """

        old_integer_list = copy.copy(self._integer_list)

        if random.random() < probability:
            self._replace_gate_with_random_gate()
        else:
            self._change_qubit_connections()
        self._fix_duplicate_qubit_assignment()
        self._update_theta_list(old_integer_list, self._integer_list)

        mutated_chromosome = copy.deepcopy(self)
        return mutated_chromosome

    def _replace_gate_with_random_gate(self) -> None:
        """
        Randomly selects a gate from the pool of all gates in the chromosome, and replaces it with a randomly generated new one.
        """
        random_index = random.randrange(0, int(self._length/3)) * 3

        self._integer_list[random_index] = random.randrange(0, len(self._gate_types))
        self._integer_list[random_index + 1] = random.randrange(0, 3) #(0, len(self._gate_types))
        self._integer_list[random_index + 2] = random.randrange(0, 3) #(0, len(self._gate_types))

    @DeprecationWarning
    def _replace_with_random_chromosome(self) -> None:
        """Clears the chromosome and randomly generates a new _integer_list"""
        gates = int(self._length/3)
        self.clear()
        self.generate_random_chromosome(gates)

    def _change_qubit_connections(self) -> None:
        """
        Finds randomly a gate and randomly changes the target qubit for single qubit gates,
        and both target and control for multiple qubit gates
        """

        random_index = random.randrange(0, int(self._length / 3)) * 3

        original_connection_1 = self._integer_list[random_index + 1]
        original_connection_2 = self._integer_list[random_index + 2]

        while original_connection_1 == self._integer_list[random_index + 1]:
            self._integer_list[random_index + 1] = random.randrange(0, 3)

        while original_connection_2 == self._integer_list[random_index + 2]:
            self._integer_list[random_index + 2] = random.randrange(0, 3)

    def _fix_duplicate_qubit_assignment(self) -> None:
        """
        Checks the chromosome for gates that connects multiple qubits.
        If the gate has an invalid connection (it is connected to itself through the randomly generated integers),
        it generates a valid configuration randomly.
        """
        gates = int(self._length / 3)

        for i in range(0, gates):
            int_index = i * 3

            if ((self._gate_dict[str(self._integer_list[int_index])] in ['cx', 'swap', 'rzz', 'rxx']) and
                    self._integer_list[int_index + 1] == self._integer_list[int_index + 2]):

                if self._integer_list[int_index + 1] == 0:
                    self._integer_list[int_index + 1] = random.randrange(1, 3)

                elif self._integer_list[int_index + 1] == 1:
                    self._integer_list[int_index + 2] = 0

                elif self._integer_list[int_index + 1] == 2:
                    self._integer_list[int_index + 2] = random.randrange(0, 2)






