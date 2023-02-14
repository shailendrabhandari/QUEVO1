import qutip
import numpy as np
import Quevo

if __name__ == '__main__':
    #mutation_prob = 10
    n_qubit = 3
    n_states = 8
    number_of_runs = 10
    gates = 10
    chromosomes = 20
    generations = 20
    gate_types = ['cx', 'x', 'h', 'rxx', 'rzz', 'swap', 'z', 'y', 'toffoli']
    states = np.random.rand(n_states, 2**3)



# Generate initial generation of chromosomes

    # Generate initial generation of chromosomes

    generation = Quevo.Generation(chromosomes, gates)
    generation.create_initial_generation(gate_types)
    #generation.run_generation(meyer_wallach_measure)
    generation.run_generation(states)

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
        generation.run_generation(states)
        current_fitness = generation.get_best_fitness()
        current_chromosome = generation.get_best_chromosome()

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

        if current_fitness < 0.01:
            break
    print("Best fitness found: " + str(final_fitness))
    print("Best chromosome found: " + str(best_chromosome))

    circuit = Quevo.Circuit(best_chromosome)
    print("\n")

    circuit.generate_circuit()
    circuit.draw()


