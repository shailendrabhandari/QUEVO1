import QUEVO
import numpy as np
if __name__ == '__main__':
    n_qubits =3
    gates = 5
    chromosomes = 20
    generations = 2
    gate_types = ['cx', 'x', 'z', 'y', 'h', 'rxx', 'rzz', 'swap', 'toffoli']
    target_entanglement = [0.9999999999999996]


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
    for i, circuit in enumerate(circuit_list):
        print(f"Circuit from parents {i+1}")
        circuit.draw()
    print("\n")


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

    # Compute the value of rho_k_sq for the best circuit
    state_vector = best_circuit.get_statevector()
    state_vector = np.reshape(state_vector, [2] * n_qubits)

    entanglement_sum = 0
    for k in range(best_circuit.n_qubits):
        rho_k_sq = np.abs(
            np.trace(np.transpose(state_vector, axes=np.roll(range(best_circuit.n_qubits), -k))) ** 2)
        entanglement_sum += rho_k_sq

    entanglement = 1 * (1 - (1 / best_circuit.n_qubits) * entanglement_sum)

    # Print the value of rho_k_sq for the best circuit
    print("Value of reduced density matrix for the best circuit:", rho_k_sq)
    print(entanglement)