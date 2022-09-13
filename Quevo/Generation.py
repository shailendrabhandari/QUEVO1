#  Copyright 2022 Sebastian T. Overskott Github link: https://github.com/Overskott/Quevo
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import math
import random
from typing import List

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

    def create_initial_generation(self, gate_types: List[str]) -> None:
        """
        Populates the generation with chromosomes.
        """
        self._chromosome_list.clear()
        for i in range(self._chromosomes):
            chromosome = Chromosome(gate_types)
            chromosome.generate_random_chromosome(self._gates)
            self._chromosome_list.append(chromosome)

    def evolve_into_next_generation(self, probability=30):
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
        self.set_parent_list()
        self._chromosome_list.clear()
        self._chromosome_list = self._parent_list.copy()

        probability_list = self.find_fitness_proportionate_probabilities()
        probability_list.reverse()

        while len(self._chromosome_list) < self._chromosomes:

            mutated_chromosome = self.select_parent(probability_list)
            mutated_chromosome.mutate_chromosome(probability)

            self._chromosome_list.append(mutated_chromosome)

    def set_parent_list(self) -> None:
        """Finds the four best chromosomes, and adds them to parent_list"""
        parent_list = self._chromosome_list.copy()
        parent_list.sort()
        self._parent_list.clear()
        self._parent_list = parent_list[:4]

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

        if math.isinf(fitness_sum):  # KL-fitness might give inf as fitness
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

            if probability < total_probability:
                parent = self._parent_list[index]
                return copy.deepcopy(parent)
            index = index + 1

    def run_generation_diff(self, desired_outcome: List[float]) -> None:
        """
        Runs the simulator for all the chromosomes in the generation and
        stores the fitness for each chromosome in fitness_list.

        Parameters
        ----------
        desired_outcome (List[float]):
            A list of the eight CA outcomes we wish to test the chromosomes against.
        """

        for chromosome in self._chromosome_list:
            circuit = Circuit(chromosome)
            chromosome_fitness = abs(circuit.find_chromosome_fitness(desired_outcome))
            chromosome.set_fitness_score(chromosome_fitness)

    def run_generation_kl(self, desired_outcome: List[float]) -> None:
        """
        Runs the simulator for all the chromosomes in the generation and
        stores the fitness for each chromosome in fitness_list.

        Parameters
        ----------
        desired_outcome (List[float]):
            A list of the eight CA outcomes we wish to test the chromosomes against.
        """

        for chromosome in self._chromosome_list:
            circuit = Circuit(chromosome)
            chromosome_fitness = abs(circuit.find_kullback_liebler_fitness(desired_outcome))
            chromosome.set_fitness_score(chromosome_fitness)

    def get_best_fitness(self):
        """Returns the fitness value for the best chromosome in the generation."""
        best_fitness = 10
        for chromosome in self._chromosome_list:
            chromosome_fitness = chromosome.get_fitness_score()
            if best_fitness > chromosome_fitness:
                best_fitness = chromosome_fitness

        return best_fitness

    def get_best_chromosome(self):
        """Returns the chromosome with the best fitness in the generation."""
        for chromosome in self._chromosome_list:
            best_fitness = self. get_best_fitness()
            if chromosome.get_fitness_score() == best_fitness:
                return chromosome
            else:
                continue

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
