'''Copyright [2023] [shailendra Bhandari]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'''

import QUEVO
import numpy as np
if __name__ == '__main__':
    n_qubits =5
    gates = 5
    chromosomes = 20
    generations = 10
    gate_types = ['cx', 'x', 'z', 'y', 'h', 'rxx', 'rzz', 'swap', 'toffoli']


# Generate initial generation of chromosomes
    generation = QUEVO.Generation(chromosomes, gates)
    generation.create_initial_generation(gate_types)
    #generation.run_generation(meyer_wallach_measure)
    generation.run_generation()
    print("Fitness for best chromosome: " + str(generation.get_best_fitness()) + "\n"
          + "Selected parents: \n")
    generation.print_parents()
    print("\n")

    # Final value placeholders
    best_circuit_state_vector = None
    best_circuit_entanglement = None

    current_chromosome = generation.get_best_chromosome()
    best_chromosome = current_chromosome
    final_fitness = generation.get_best_fitness()

    # Mutation loop
    for gen in range(0, generations):

        # Mutate next generation of chromosomes
        generation.evolve_into_next_generation()
        # Check every Chromosome's fitness
        generation.run_generation()
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
        print(f"Circuit from parents {i+1}")
        circuit.draw()
    print("\n")'''


 # Check if there is a new best chromosome
    if best_circuit_entanglement is None or best_fitness > best_circuit_entanglement:
        # Store the best circuit state vector and entanglement
        best_circuit = QUEVO.Circuit(best_chromosome)
        best_circuit.generate_circuit()
        best_circuit_state_vector = best_circuit.get_statevector()
        best_circuit_entanglement = best_fitness

    # Print the best circuit
    print("Best circuit:", best_circuit_entanglement)
    best_circuit.draw()

    # Print the state vector of the best circuit
    print("State vector of the best circuit:")
    print(best_circuit_state_vector)

