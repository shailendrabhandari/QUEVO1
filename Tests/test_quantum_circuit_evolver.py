from unittest import TestCase
from quantum_circuit_evolver import *


class TestCircuitString(TestCase):

    def test_check_duplicate_qubit_assignment(self):
        assert True

    def test_generate_gate_string(self):
        assert True

    def test_mutate_gate_string(self):
        assert True

    def test_set_gate_string(self):
        assert True

    def test_get_gates_string(self):
        test_circuit_string = Chromosome()
        assert test_circuit_string.get_integer_list() == []

        int_list = [1, 2, 3]
        test_circuit_string.set_integer_list(int_list)
        assert test_circuit_string.get_integer_list() == int_list

    def test_clear_string(self):
        test_circuit_string = Chromosome()

        test_circuit_string.set_integer_list([1, 2, 3])
        assert test_circuit_string.clear() is None


class TestCircuitGenerator:
    def test_generate_circuit(self):
        assert True

    def test_initialize_initial_states(self):
        assert True

    def test_create_initial_generation(self):
        assert True

    def test_run_generation(self):
        assert True

    def test_calculate_error(self):
        assert True

    def test_run_circuit(self):
        assert True

    def test_find_chromosome_fitness(self):
        assert True

    def test_draw_circuit(self):
        assert True

    def test_set_gate_string(self):
        assert True

    def test_get_gates_string(self):
        assert True

    def test_clear_circuit(self):
        assert True

    def test_clear_string(self):
        assert True
