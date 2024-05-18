import math
from tqdm import tqdm
import numpy as np

from composite_ms.qubit_operator import QubitOperator

np.random.seed(0)


class DerandomizationMeasurement:
    def __init__(self, operator: QubitOperator, use_weight=True):
        """
        Args:
            operator: terms in the Hamiltonian
            nqubit: num of qubits
            use_weight: whether we use the values of the coefficients in the Hamiltonian
        """
        self.ops = operator
        self.nqubit = operator.n_qubit
        self.weights, self.pwords,  = operator.get_pauli_tensor()
        self.weights = np.abs(self.weights)
        if use_weight:
            self.weights = self.weights / np.max(self.weights)
        else:
            self.weights = np.array([1 for _ in range(len(self.terms))])

    def build(self, min_nshot_a_term, max_nshot=None):
        """
        Args:
            min_nshot_a_term: number of measurements
            nqubit: number of qubits

        Returns: an array of Pauli strings
        """
        max_nshot = max_nshot or min_nshot_a_term * len(self.pwords) // 5
        return self._derandomized_classical_shadow(self.pwords, min_nshot_a_term, max_nshot)
        

    def get_cost_function(self):
        cost_config = DerandomizationCost(self.nqubit, self.weights, shift=0)

        def cost_fun(hit_counts, not_matched_counts):
            V = cost_config.eta / 2 * hit_counts
            if_nqubit_smaller_than_maches_needed = np.heaviside(cost_config.nqubit - not_matched_counts, 1)
            V += if_nqubit_smaller_than_maches_needed * (- np.log(1 - cost_config.nu / (3 ** not_matched_counts)))
            cost_val = np.sum(np.exp(-V / np.array(self.weights)))
            return cost_val

        return cost_fun

    def _derandomized_classical_shadow(self, observables, min_nshot_a_term, max_nshot):
        """
        Forked from
        https://github.com/hsinyuan-huang/predicting-quantum-properties/blob/master/data_acquisition_shadow.py
        and updated
        """
        hit_counts = np.array([0] * len(observables))
        results = []
        cost_fun = self.get_cost_function()
        self.weights = np.array(self.weights)
        with tqdm(range(max_nshot), ncols=100) as pbar:
            for n_step in pbar:
                # A single round of parallel measurement over "system_size" number of qubits
                not_matched_counts = np.sum(np.sign(observables), axis=-1)  # np.array([len(P) for P in observables])
                # Measurement Pauli string
                measurement = []
                for qubit_index in range(self.nqubit):
                    candidate_matched_counts_for_best_pauli = None
                    ps = [1, 2, 3]
                    np.random.shuffle(ps)
                    best_pauli = -1
                    min_cost = np.inf
                    for pauli_candidate in ps:
                        candidate_matched_counts = not_matched_counts.copy()
                        observables_on_qubit = observables[:, qubit_index]
                        matched_observables = 1 * (observables_on_qubit == pauli_candidate)
                        mis_matched_observables = 1 * np.logical_and(observables_on_qubit != pauli_candidate, observables_on_qubit != 0)
                        candidate_matched_counts += 100 * (self.nqubit + 10) * mis_matched_observables
                        candidate_matched_counts -= matched_observables
                        cost_for_pauli = cost_fun(hit_counts, candidate_matched_counts)
                        if cost_for_pauli < min_cost:
                            best_pauli = pauli_candidate
                            min_cost = cost_for_pauli
                            candidate_matched_counts_for_best_pauli = candidate_matched_counts
                    not_matched_counts = candidate_matched_counts_for_best_pauli
                    measurement.append(best_pauli)
                results.append(measurement)
                hit_counts += 1 - np.sign(np.abs(not_matched_counts))
                if min_nshot_a_term != -1:
                    least_satisfied = np.min(hit_counts - np.floor(min_nshot_a_term*self.weights))
                    pbar.set_description(str(least_satisfied))
                    if least_satisfied >= 0:
                        break
        return results


class DerandomizationCost:
    def __init__(self, nqubit, weights, shift):
        self.weights = weights
        self.nqubit = nqubit
        self.shift = shift
        self.sum_log_value = 0
        self.sum_cnt = 0
        self.eta = 0.9
        self.nu = 1 - math.exp(-self.eta / 2)