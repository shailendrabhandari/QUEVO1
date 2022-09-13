
import Quevo

if __name__ == '__main__':
    mutation_prob = 10
    number_of_runs = 10
    gates = 15
    chromosomes = 20
    generations = 150
    gate_types = ['cx', 'x', 'h', 'rxx', 'rzz', 'swap', 'z', 'y', 'toffoli']

    
    #desired_chance_of_one = [0.394221, 0.094721, 0.239492, 0.408455, 0.0, 0.730203, 0.915034, 1.0]
    # Probabilities from : https://link.springer.com/article/10.1007/s11571-020-09600-x
    desired_chance_of_one = [0, 1, 1, 1, 0, 1, 1, 0] # rule 110 of cellular automata 
    #desired_chance_of_one = [0, 1, 0, 1, 1, 0, 1, 0] # rule 90 of the cellular automata
    #desired_chance_of_one = [0.6363719194178489, 0.6603379889493801, 0.5261171737956705,0.1747797354154147,     0.881963735323847, 0.33714054950294126, 0.03402241626006142, 0.44436322091652825] #random1
    #desired_chance_of_one = [0.4778229501941019, 0.5603843573904608, 0.8528375946918625, 0.48177324440790203,       0.31425559382377966, 0.3463862375593587, 0.06782268262323932, 0.9123753856359555] #random2
    #desired_chance_of_one = [0.198767809319908, 0.4701204970147985, 0.9836406277843224,0.7114722334790647,  0.6615993900216182, 0.12184009533885554,0.1327895891110945, 0.7306276574780672]#random3
     


# Generate initial generation of chromosomes

    # Generate initial generation of chromosomes

    generation = Quevo.Generation(10, gates)
    generation.create_initial_generation(gate_types)
    generation.run_generation_kl(desired_chance_of_one)

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
        generation.run_generation_kl(desired_chance_of_one)

        current_fitness = generation.get_best_fitness()
        current_chromosome = generation.get_best_chromosome()

        # Print generation best result
        print("Fitness for best mutated chromosome in mutation " + str(gen + 1) + ": "
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
    circuit.print_ca_outcomes(desired_chance_of_one)
    print("\n")

    circuit.generate_circuit()
    circuit.draw()


