
from tqdm import tqdm
import numpy as np

from composite_ms.qubit_operator import QubitOperator

np.random.seed(0)


class ShadowGroupingMeasurement:
    def __init__(self, operator: QubitOperator, use_weight=True):
        """
        Args:
            operator: terms in the Hamiltonian
            nqubit: num of qubits
            use_weight: whether we use the values of the coefficients in the Hamiltonian
        """
        self.ops = operator
        self.nqubit = operator.n_qubit
        self.coeffs, self.pwords,  = operator.get_pauli_tensor()
        self.coeffs = np.abs(self.coeffs)
        # self.weight_function = Derandomization_weight(focus_on_greedy=False)
        self.weight_function = Energy_estimation_inconfidence_weight()
        self.eps = 1e-4
        if use_weight:
            self.coeffs = self.coeffs / np.max(self.coeffs)  # [abs(t.amp) / max_weight for t in self.terms]
        else:
            self.coeffs = np.array([1 for _ in range(len(self.pwords))])


    def build(self, min_nshot_a_term, max_nshot=None):
        """
        Args:
            max_nshot: maximal number of measurements
            nqubit: number of qubits

        Returns: an array of Pauli strings
        """
        max_nshot = max_nshot or min_nshot_a_term * len(self.pwords) // 5
        return self._derandomized_classical_shadow_vectorized(self.pwords, min_nshot_a_term, max_nshot)


    def _derandomized_classical_shadow_vectorized(self, observables, min_nshot_a_term, max_nshot):
        """
        Refered to
        https://github.com/hsinyuan-huang/predicting-quantum-properties/blob/master/data_acquisition_shadow.py
        """
        hit_counts = np.array([0] * len(observables))
        results = []
        self.coeffs = np.array(self.coeffs)
        with tqdm(range(max_nshot), ncols=100) as pbar:
            for n_step in pbar:
                weights = self.weight_function.get_weights(self.coeffs, self.eps, hit_counts)
                order = np.argsort(weights)
                # Measurement Pauli string
                measurement = np.zeros(self.nqubit)
                for idx in reversed(order):
                    pword = self.pwords[idx]
                    hit = (measurement == pword) | (measurement == 0) | (pword == 0)
                    hit = hit.all()
                    if hit:
                        non_id = pword != 0
                        # overwrite those qubits that fall in the support of o
                        measurement[non_id] = pword[non_id]
                        # break sequence is case all identities in setting are exhausted
                        if np.min(measurement) > 0:
                            break
                is_hit = (self.pwords == measurement) | (self.pwords == 0)
                is_hit = 1 * is_hit.all(axis=-1)
                hit_counts += is_hit
                results.append(measurement)
                
                if min_nshot_a_term != -1:
                    least_satisfied = np.min(hit_counts - np.floor(min_nshot_a_term * self.coeffs))
                    pbar.set_description(str(least_satisfied))
                    if least_satisfied >= 0:
                        break

        return results


class Derandomization_weight():
    def __init__(self, focus_on_greedy=True):
        self.greedy = focus_on_greedy
        return

    def get_weights(self, w, eps, N_hits):
        inconf = np.exp(-0.5 * eps * eps * N_hits / (w ** 2))
        inconf -= np.exp(-0.5 * eps * eps * (N_hits + 1) / (w ** 2))
        if self.greedy:
            inconf[N_hits == 0] -= 1
            inconf[N_hits == 0] *= -1
        return inconf

    def __call__(self):
        return self.get_weights

    
class Energy_estimation_inconfidence_weight():
    def __init__(self, alpha=1):
        self.alpha = alpha
        assert alpha >= 1, "alpha has to be chosen larger or equal 1, but was {}.".format(alpha)
        return

    def get_weights_for_testing(self, w, eps, N_hits):
        inconf = self.alpha * w ** 2
        condition = N_hits != 0
        inconf[condition] /= self.alpha * (N_hits[condition] + 1) * N_hits[condition]
        return inconf

    def get_weights(self, w, eps, N_hits):
        inconf = self.alpha * np.abs(w)
        condition = N_hits != 0
        N = np.sqrt(N_hits[condition])
        Nplus1 = np.sqrt(N_hits[condition] + 1)
        inconf[condition] /= self.alpha * np.sqrt(N * Nplus1) / (Nplus1 - N)
        return inconf

    def __call__(self):
        return self.get_weights