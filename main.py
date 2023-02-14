import qutip
import numpy as np
import QUEVO
import itertools
if __name__ == '__main__':
    #mutation_prob = 10
    n_qubits = 3
    n_states = 8
    number_of_runs = 1
    gates = 10
    chromosomes = 10
    generations = 10
    gate_types = ['cx', 'x', 'h', 'rxx', 'rzz', 'swap', 'z', 'y', 'toffoli']
    #states = [1, 1, 1, 1, 1, 1, 1, 1]
    states = np.random.rand(n_states, 2**n_qubits)
    #states = list(itertools.product(qubit_states, repeat=3))
    '''For n_states = 8 and n_qubits = 3, the above code will create
       a 2D numpy array with 8 rows and 2^3=8 columns, where each element 
       in the array is a random number between 0 and 1. Eg.
       [[0.16248639, 0.43624932, 0.51693115, 0.76832434, 0.70870195, 0.47029482, 0.04004524, 0.39170316],
       [0.12674473, 0.7255842 , 0.36640728, 0.81731192, 0.72618788, 0.7491775 , 0.64095472, 0.03906416],
       [0.58217411, 0.10484563, 0.88036448, 0.25417107, 0.56672723, 0.81278999, 0.89182192, 0.30158468],
       [0.40554661, 0.67279258, 0.98864606, 0.43253257, 0.96366368, 0.22205014, 0.11458701, 0.90781098],
       [0.56837468, 0.5898366 , 0.30850198, 0.47076449, 0.87769243, 0.14045381, 0.68710867, 0.36902781],
       [0.15074584, 0.86403677, 0.59870491, 0.2342205 , 0.53060639, 0.77072021, 0.59867769, 0.73537197],
       [0.76457804, 0.30510915, 0.14162209, 0.3195406 , 0.19285752, 0.93230133, 0.91168392, 0.60262369],
       [0.3564249 , 0.32999487, 0.52168684, 0.92656499, 0.30701791, 0.33491614, 0.24813214, 0.20212784]])
     '''


# Generate initial generation of chromosomes

    # Generate initial generation of chromosomes

    generation = QUEVO.Generation(chromosomes, gates)
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
    print("Best fitness for the circuit: " + str(final_fitness))
    print("Best chromosome : " + str(best_chromosome))

    circuit = Quevo.Circuit(best_chromosome)
    print("\n")

    circuit.generate_circuit()
    circuit.draw()


