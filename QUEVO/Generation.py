
import copy
import math
import random
from typing import List
import numpy as np
from .Chromosome import Chromosome
from .Circuit import Circuit


class Generation(object):
    """
    This class represents a collection of chromosomes. The Collection is called a
    generation.

    Attributes
    ----------
    _chromosome_list: List[int]
        List of chromosomes that habits the generation.
    _parent_list: List[Chromosome]
        list of the chromosomes that are chosen to be parents in the generation.
    _chromosomes: int
        The number of chromosomes in the generation
    _gates: int
        Number of gates in each chromosome.
    """

    def __init__(self, chromosomes: int, gates: int) -> None:
        """
        The Generation constructor.

        Parameters
        ----------
        chromosomes (int):
            Number of chromosomes in the generation.
        gates (int):
            Number of gates in each chromosome.
        """
        self._chromosome_list: List[Chromosome] = []
        self._parent_list: List[Chromosome] = []
        self._chromosomes: int = chromosomes
        self._gates: int = gates
        #self.mutation_rate: float = mutation_rate



        self._chromosome_list = []
        self._parent_chromosomes = []
        self._circuit_list = []  # added line
        self._circuit_fitness_list = []  # added line
        #self._mutation_rate = mutation_rate
        self._best_fitness = None
        self._best_chromosome = None

    def create_initial_generation(self, gate_types: List[str]) -> None:
        """
        Populates the generation with chromosomes.
        """
        self._chromosome_list.clear()
        for i in range(self._chromosomes):
            chromosome = Chromosome(gate_types)
            chromosome.generate_random_chromosome(self._gates)
            self._chromosome_list.append(chromosome)
        # Print the generated chromosomes
        #print("Initial Generation Parents:")
        '''for chromosome in self._chromosome_list:
            print(chromosome)'''

    def evolve_into_next_generation(self, probability=0.05):
        """
        Changes the chromosomes in the generation by "evolving" them is this manner:
        The four best chromosomes are left unchanged as "elites". the rest of the chromosomes are
        evolved with the mutate_chromosome() function.

        Parameters
        ----------
        [Optional] probability (int)
            The probability for the mutation to be replaced a random gate with a random new one. The chance of
            mutating by changing a gate connection(s) is (1-probability).
        """
        #print(f"Mutation probability: {probability}")  # Print the mutation probability
        self.set_parent_list()
        new_chromosome_list = []

        # add elite chromosomes to the chromosome list
        for elite_chromosome in self._parent_list:
            new_chromosome_list.append(elite_chromosome)

        # create new chromosomes using parent selection and mutation
        probability_list = self.find_fitness_proportionate_probabilities()
        probability_list.reverse()

        while len(new_chromosome_list) < self._chromosomes:
            mutated_chromosome = self.select_parent(probability_list)
            mutated_chromosome.mutate_chromosome(probability)
            new_chromosome_list.append(mutated_chromosome)

        self._chromosome_list = new_chromosome_list


    def set_parent_list(self) -> None:
        """Finds the four best chromosomes, and adds them to parent_list"""
        sorted_chromosomes = sorted(self._chromosome_list, key=lambda c: np.mean(np.abs(c.get_fitness_score())), reverse=True)
        self._parent_list = sorted_chromosomes[:4]

    def find_fitness_proportionate_probabilities(self) -> List[float]:
        """
        Calculates and returns the fitness proportionate probabilities for the four parent in the generation.

        Returns
        -------
        selection_list (List[float])
            A list describing the fitness proportionate probabilities for the four parent in the generation.
        """
        fitness_sum = 0
        for parent in self._parent_list:
            fitness_sum = fitness_sum + parent.get_fitness_score()

        if np.isinf(fitness_sum).any():
            fitness_sum = 100

        selection_list = []
        for parent in self._parent_list:
            selection_probability = (parent.get_fitness_score()/fitness_sum)
            selection_list.append(selection_probability)

        selection_list.reverse()
        return selection_list

    def select_parent(self, probability_list) -> Chromosome:
        """
        Selects one of the four parents based on the fitness proportionate probabilities and returns that parent.

        Parameters
        ----------
        probability_list
            A list of the probabilities for parent selection. should add up to 1.

        Returns
        -------
        Parent (Chromosome)
            The chosen parent
        """
        probability = random.uniform(0, 1)

        total_probability = 0
        index = 0

        for prob in probability_list:
            total_probability = total_probability + prob

            if np.any(probability < total_probability):
                parent = self._parent_list[index]
                return copy.deepcopy(parent)
            index = index + 1

    #def run_generation(self, states: np.ndarray)->None:
    def run_generation(self, target_entanglement: List[float]) -> None:
        circuit_list = []  # added line
        circuit_fitness_list = []  # added line

        for chromosome in self._chromosome_list:
            circuit = Circuit(chromosome)
            circuit.generate_circuit()
            fitness = circuit.find_chromosome_fitness(target_entanglement)
            circuit_list.append(circuit)  # added line
            circuit_fitness_list.append(fitness)  # added line
            chromosome.set_fitness_score(fitness)
        self._circuit_list.append(circuit_list)  # added line
        self._circuit_fitness_list.append(circuit_fitness_list)  # added line

    def get_circuit_list(self, generation_index: int) -> List[Circuit]:
        return self._circuit_list[generation_index]

    def get_circuit_fitness_list(self, generation_index: int) -> List[float]:
        return self._circuit_fitness_list[generation_index]

    def get_best_fitness(self) -> float:
        """
        Returns the fitness of the best chromosome in the generation.

        Returns
        -------
        best_fitness (float)
            The fitness of the best chromosome in the generation.
        """
        best_fitness = float("inf")
        for chromosome in self._chromosome_list:
            chromosome_fitness = abs(chromosome.get_fitness_score())
            if np.max(chromosome_fitness) < best_fitness:
                best_fitness = np.max(chromosome_fitness)
        return best_fitness

    def get_best_chromosome(self) -> Chromosome:
        """
        Returns the best chromosome in the generation.

        Returns
        -------
        best_chromosome (Chromosome)
            The best chromosome in the generation.
        """
        best_fitness = float("inf")
        best_chromosome = None
        for chromosome in self._chromosome_list:
            chromosome_fitness = abs(chromosome.get_fitness_score()).max()
            if chromosome_fitness < best_fitness:
                best_fitness = chromosome_fitness
                best_chromosome = chromosome
        return best_chromosome

    def print_chromosomes(self):
        """Prints all the generation's chromosomes."""
        print("Chromosomes: ")
        for chromosome in self._chromosome_list:
            print(chromosome)
        print('\n')

    def print_theta_values(self):
        """Prints all the generation's theta values."""
        print("Theta values: ")
        for chromosome in self._chromosome_list:
            print(chromosome.get_theta_list())
        print('\n')

    def print_parents(self):
        """Prints the generation's selected parents."""
        if not self._parent_list:
            print("No parents selected")
        else:
            print("Parents: ")
            for parent in self._parent_list:
                print(parent)
            print('\n')
    def get_best_circuit(self):
        """Returns the circuit with the best fitness in the generation."""
        best_fitness = self.get_best_fitness()
        for chromosome in self._chromosome_list:
            if chromosome.get_fitness_score() == best_fitness:
                circuit = Circuit(chromosome)
                circuit.generate_circuit()
                return (circuit, best_fitness)
        return None


    def print_circuits(self):
        """Prints all the generation's chromosome's circuits."""
        print("Circuits: ")
        for chromosome in self._chromosome_list:
            circuit = Circuit(chromosome)
            circuit.generate_circuit()
            circuit.draw()
        print("\n")

    def print_fitness(self):
        """Prints the generation's chromosome's fitness"""
        for chromosome in self._chromosome_list:
            print(chromosome.get_fitness_score())
        print("\n")




