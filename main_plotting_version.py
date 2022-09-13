import Quevo
import matplotlib.pyplot as plt
mutation_prob = 5
number_of_runs= 10
gates = 20
chromosomes = 20
generations = 100
gate_types = ['cx', 'x', 'h', 'rxx', 'rzz', 'swap', 'z', 'y', 'toffoli']
# possible gates: # h, cx, x, swap, rzz, rxx, toffoli, y, z

desired_chance_of_one = [0.394221, 0.094721, 0.239492, 0.408455, 0.0, 0.730203, 0.915034, 1.0]
# Generate initial generation of chromosomes
def run_evolution(number_of_runs, probability=30):
   # plot_list = []

    for i in range(0, number_of_runs):

        print('Evolution number: ' + str(i))
        init_gen = Quevo.Generation(chromosomes, gates)
        init_gen.create_initial_generation(gate_types)

        init_gen.run_generation_kl(desired_chance_of_one)

        # Final value placeholders
        current_chromosome = init_gen.get_best_chromosome()
        best_chromosome = current_chromosome
        final_fitness = init_gen.get_best_fitness()

        final_fitness_list = [final_fitness]

        # Mutation loop

        for gen in range(0, generations):
            print('Running gen nr.' + str(gen + 1))
            # Mutate next generation of chromosomes
            init_gen.evolve_into_next_generation(probability)  # Add probability here!

            # Check every Chromosome's fitness
            init_gen.run_generation_kl(desired_chance_of_one)

            current_fitness = init_gen.get_best_fitness()
            current_chromosome = init_gen.get_best_chromosome()

            # Check if there is a new_list best chromosome

            if final_fitness > abs(current_fitness):
                final_fitness = current_fitness
                best_chromosome = current_chromosome

            final_fitness_list.append(current_fitness)
            if current_fitness < 0.01:
                break

    return \
        final_fitness_list


# edited section
'''def mutation_prob(mutation_prob,probability=30):
    for i in range(0,mutation_prob):
        print('Mutation Prob'+str(i))
        init_gen.evolve_into_next_generation(probability) # Add probability here!

        # Check every Chromosome's fitness
        init_gen.run_generation_kl(desired_chance_of_one)

        current_fitness = init_gen.get_best_fitness()
        current_chromosome = init_gen.get_best_chromosome()


            # Check if there is a new_list best chromosome

        if final_fitness > abs(current_fitness):
            final_fitness = current_fitness
            best_chromosome = current_chromosome

            final_fitness_list.append(current_fitness)
        if current_fitness < 0.01:
            break


    return final_fitness_list'''


#For plotting No of generation Vs fitness score
'''plt.plot(run_evolution(1, 0), label="1st")
plt.plot(run_evolution(1, 35), label="2nd")
plt.plot(run_evolution(1, 50), label="3rd")
plt.plot(run_evolution(1, 70), label="4th")
plt.plot(run_evolution(1, 90), label="5th")
plt.plot(run_evolution(1, 105), label="6th")
plt.plot(run_evolution(1, 120), label="7th")
plt.plot(run_evolution(1, 135), label="8th")
plt.plot(run_evolution(1, 150), label="9th")'''
'''plt.plot(run_evolution(1, 90), label="10th")
plt.plot(run_evolution(1, 100), label="11th")
plt.plot(run_evolution(1, 110), label="12th")
plt.plot(run_evolution(1, 120), label="13th")
plt.plot(run_evolution(1, 130), label="14th")
plt.plot(run_evolution(1, 140), label="15th")
plt.plot(run_evolution(1, 150), label="16th")

plt.xlabel("Number of generations")
plt.ylabel("Fitness score")
# plt.title("Circuit fitness over generations")
plt.legend()
plt.savefig('20diff_gates_30%mutation.pdf')
plt.show()'''

 #plot_list = []
for i in range(0,5):
    init_gen = Quevo.Generation(chromosomes, gates) # Number of gates in chromosome:3,5,10,15,20
    init_gen.create_initial_generation(gate_types)

    init_gen.run_generation_diff(desired_chance_of_one)  #init_gen.run_generation_kl >> for kl
                                                       #init_gen.run_generation_diff >>for diff

    print("Fitness for best chromosome: " + str(init_gen.get_best_fitness()) + "\n"
          + "Selected parents: \n")
    init_gen.print_parents()
    print("\n")

    # Final value placeholders
    current_chromosome = init_gen.get_best_chromosome()
    best_chromosome = current_chromosome
    final_fitness = init_gen.get_best_fitness()

    final_fitness_list = [final_fitness]

    # Mutation loop

    for gen in range(0, generations):

        # Mutate next generation of chromosomes
        init_gen.evolve_into_next_generation() # Add mutation probability here! the default is 10% for exp1.

        # Check every Chromosome's fitness
        init_gen.run_generation_diff(desired_chance_of_one) #init_gen.run_generation_kl >> for kl
                                                            #init_gen.run_generation_diff >>for diff
        current_fitness = init_gen.get_best_fitness()
        current_chromosome = init_gen.get_best_chromosome()

        # Print generation best result
        print("Fitness for best mutated chromosome in mutation " + str(gen + 1) + ": "
              + str(current_fitness) + "\n")
        init_gen.print_parents()
        print("------------------------------------------------------------------------------")
        #print("\n")

        # Check if there is a new_list best chromosome

        if final_fitness > abs(current_fitness):
            final_fitness = current_fitness
            best_chromosome = current_chromosome
            print("New best!")
            print("------------------------------------------------------------------------------")
            print("\n")

        final_fitness_list.append(current_fitness)
        if current_fitness < 0.01:
            break
    print("Best fitness found: " + str(final_fitness))
    print("Best chromosome found: " + str(best_chromosome))

    circuit = Quevo.Circuit(best_chromosome)
    circuit.print_ca_outcomes(desired_chance_of_one)
    print("\n")

    circuit.generate_circuit()
    circuit.draw()

   # plot_list.append(final_fitness_list)