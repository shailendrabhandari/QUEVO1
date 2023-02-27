import qutip
from qiskit import QuantumCircuit
import numpy as np
import QUEVO
from math import ceil
import itertools
if __name__ == '__main__':
    #mutation_prob = 10
    number_of_runs = 100
    gates = 10
    chromosomes = 40
    generations = 20
    gate_types = ['cx', 'x', 'h', 'rxx', 'rzz', 'swap', 'z', 'y', 'toffoli']
    target_entanglement = [0.999999,0.999999]


# Generate initial generation of chromosomes

    # Generate initial generation of chromosomes

    generation = QUEVO.Generation(chromosomes, gates, mutation_rate=0.02)
    generation.create_initial_generation(gate_types)
    #generation.run_generation(meyer_wallach_measure)
    generation.run_generation(target_entanglement)



    print("Fitness for best chromosome: " + str(generation.get_best_fitness()) + "\n"
          + "Selected parents: \n")
    generation.print_parents()
    print("\n")

    # Final value placeholders
    current_chromosome = generation.get_best_chromosome()
    best_chromosome = current_chromosome
    final_fitness = generation.get_best_fitness()

    # Mutation loop
    for gen in range(0, generations):

        # Mutate next generation of chromosomes
        generation.evolve_into_next_generation()
        # Check every Chromosome's fitness
        generation.run_generation(target_entanglement)
        current_fitness = generation.get_best_fitness()
        current_chromosome = generation.get_best_chromosome()

        mean_fitness = np.mean(current_fitness)
        max_fitness = np.max(current_fitness)
        std_fitness = np.std(current_fitness)

        print("Best fitness = " + str(max_fitness))

        # Print generation best result
        print("Fitness for best mutated circuit " + str(gen + 1) + ": "
              + str(current_fitness) + "\n")
        generation.print_parents()
        print("------------------------------------------------------------------------------")
        print("\n")

        # Check if there is a new_list best chromosome
        if final_fitness > abs(current_fitness):
            final_fitness = current_fitness
            best_chromosome = current_chromosome
            print("New best!")
            print("------------------------------------------------------------------------------")
            print("\n")

        if current_fitness < 0.0001:
            break
    print("Best fitness for the circuit: " + str(final_fitness))
    print("Best chromosome : " + str(best_chromosome))

# Print the best circuit and its fitness
    best_fitness = generation.get_best_fitness()
    best_chromosome = generation.get_best_chromosome()
    print("Best fitness for generation", gen+1, ":", best_fitness)
    circuit = QUEVO.Circuit(best_chromosome)
    circuit_list = generation.get_circuit_list(gen)
    for i, circuit in enumerate(circuit_list):
        print(f"Circuit {i + 1} from generation {gen + 1}:")
        circuit.draw()
        print("\n")
    print("\n")

    circuit.generate_circuit()
    circuit.draw()


