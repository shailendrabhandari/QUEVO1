import QUEVO
if __name__ == '__main__':
    gates = 10
    chromosomes = 20
    generations = 20
    gate_types = ['cx', 'x', 'z', 'y', 'h', 'rxx', 'rzz', 'swap', 'toffoli']
    target_entanglement = [0.9999999999]


# Generate initial generation of chromosomes
    generation = QUEVO.Generation(chromosomes, gates)
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

        # Print generation best result
        print("Fitness for best mutated chromosome in generation" + str(gen + 1) + ": "
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
    # Print the best circuit and its fitness
    best_fitness = generation.get_best_fitness()
    best_chromosome = generation.get_best_chromosome()
    circuit = QUEVO.Circuit(best_chromosome)
    circuit_list = generation.get_circuit_list(gen)
    '''for i, circuit in enumerate(circuit_list):
        print(f"Circuit {i + 1} from generation {gen + 1}:")
        circuit.draw()
    print("\n")'''

    circuit.generate_circuit()
    circuit.draw()

