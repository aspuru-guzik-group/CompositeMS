#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""The file is modified by Zijian Zhang for the Mizore project"""
from __future__ import annotations
import pickle
from collections import defaultdict
from pathlib import Path

"""SymbolicOperator is the base class for FermionOperator and QubitOperator"""

import abc
import copy
import itertools
import re
import warnings

import sympy

EQ_TOLERANCE = 1e-8

COEFFICIENT_TYPES = (int, float, complex, sympy.Expr)


class SymbolicOperator(metaclass=abc.ABCMeta):
    """Base class for FermionOperator and QubitOperator.

    A SymbolicOperator stores an object which represents a weighted
    sum of terms; each term is a product of individual factors
    of the form (`index`, `action`), where `index` is a nonnegative integer
    and the possible values for `action` are determined by the subclass.
    For instance, for the subclass FermionOperator, `action` can be 1 or 0,
    indicating raising or lowering, and for QubitOperator, `action` is from
    the set {'X', 'Y', 'Z'}.
    The coefficients of the terms are stored in a dictionary whose
    keys are the terms.
    SymbolicOperators of the same type can be added or multiplied together.

    Note:
        Adding SymbolicOperators is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a SymbolicOperator
        with a scalar.

    Attributes:
        actions (tuple): A tuple of objects representing the possible actions.
            e.g. for FermionOperator, this is (1, 0).
        action_strings (tuple): A tuple of string representations of actions.
            These should be in one-to-one correspondence with actions and
            listed in the same order.
            e.g. for FermionOperator, this is ('^', '').
        action_before_index (bool): A boolean indicating whether in string
            representations, the action should come before the index.
        different_indices_commute (bool): A boolean indicating whether
            factors acting on different indices commute.
        terms (dict):
            **key** (tuple of tuples): A dictionary storing the coefficients
            of the terms in the operator. The keys are the terms.
            A term is a product of individual factors; each factor is
            represented by a tuple of the form (`index`, `action`), and
            these tuples are collected into a larger tuple which represents
            the term as the product of its factors.
    """

    @staticmethod
    def _issmall(val, tol=EQ_TOLERANCE):
        '''Checks whether a value is near-zero

        Parses the allowed coefficients above for near-zero tests.

        Args:
            val (COEFFICIENT_TYPES) -- the value to be tested
            tol (float) -- tolerance for inequality
        '''
        if isinstance(val, sympy.Expr):
            if sympy.simplify(abs(val) < tol) == True:
                return True
            return False
        if abs(val) < tol:
            return True
        return False

    @abc.abstractproperty
    def actions(self):
        """The allowed actions.

        Returns a tuple of objects representing the possible actions.
        """
        pass

    @abc.abstractproperty
    def action_strings(self):
        """The string representations of the allowed actions.

        Returns a tuple containing string representations of the possible
        actions, in the same order as the `actions` property.
        """
        pass

    @abc.abstractproperty
    def action_before_index(self):
        """Whether action comes before index in string representations.

        Example: For QubitOperator, the actions are ('X', 'Y', 'Z') and
        the string representations look something like 'X0 Z2 Y3'. So the
        action comes before the index, and this function should return True.
        For FermionOperator, the string representations look like
        '0^ 1 2^ 3'. The action comes after the index, so this function
        should return False.
        """
        pass

    @abc.abstractproperty
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        pass

    def __init__(self, term=None, coefficient=1., n_site=-1):
        if not isinstance(coefficient, COEFFICIENT_TYPES):
            raise ValueError(
                'Coefficient must be a numeric type. Got {}'.format(
                    type(coefficient)))

        # Initialize the terms dictionary
        self.terms = {}

        # Detect if the input is the string representation of a sum of terms;
        # if so, initialization needs to be handled differently
        if isinstance(term, str) and '[' in term:
            self._long_string_init(term, coefficient)
            return

        # Zero operator: leave the terms dictionary empty
        if term is None:
            return

        # Parse the term
        # Sequence input
        if isinstance(term, (list, tuple)):
            term = self._parse_sequence(term)
        # String input
        elif isinstance(term, str):
            term = self._parse_string(term)
        # Invalid input type
        else:
            raise ValueError('term specified incorrectly.')

        # Simplify the term
        coefficient, term = self._simplify(term, coefficient=coefficient)

        # Add the term to the dictionary
        self.terms[term] = coefficient

        self.hash_cache = None

        self.n_site = n_site

    def get_data_dict(self):
        data = {
            "n_site": self.n_site,
            "term": self.terms
        }
        return data

    @classmethod
    def from_data_dict(cls, data):
        op = cls.from_terms_dict(data["term"])
        op.n_site = data["n_site"]
        return op

    def save_to_op_file(self, title, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        with open(folder_path + f"/{title}.op", "wb") as f:
            pickle.dump(self.get_data_dict(), f)

    @classmethod
    def read_op_file(cls, title, folder_path):
        with open(folder_path + f"/{title}.op", "rb") as f:
            data = pickle.load(f)
        return cls.from_data_dict(data)

    @classmethod
    def from_terms_dict(cls, terms):
        op = cls()
        cls.__init__(op)
        op.terms = terms
        return op

    def __hash__(self):
        # if self.hash_cache is not None:
        #    return self.hash_cache
        terms = tuple((key, value) for key, value in self.terms.items())
        res = terms.__hash__()
        self.hash_cache = res
        return res

    def _long_string_init(self, long_string, coefficient):
        r"""
        Initialization from a long string representation.

        e.g. For FermionOperator:
            '1.5 [2^ 3] + 1.4 [3^ 0]'
        """

        pattern = r'(.*?)\[(.*?)\]'  # regex for a term
        for match in re.findall(pattern, long_string, flags=re.DOTALL):

            # Determine the coefficient for this term
            coef_string = re.sub(r"\s+", "", match[0])
            if coef_string and coef_string[0] == '+':
                coef_string = coef_string[1:].strip()
            if coef_string == '':
                coef = 1.0
            elif coef_string == '-':
                coef = -1.0
            else:
                try:
                    if 'j' in coef_string:
                        if coef_string[0] == '-':
                            coef = -complex(coef_string[1:])
                        else:
                            coef = complex(coef_string)
                    else:
                        coef = float(coef_string)
                except ValueError:
                    raise ValueError(
                        'Invalid coefficient {}.'.format(coef_string))
            coef *= coefficient

            # Parse the term, simpify it and add to the dict
            term = self._parse_string(match[1])
            coef, term = self._simplify(term, coefficient=coef)
            if term not in self.terms:
                self.terms[term] = coef
            else:
                self.terms[term] += coef

    def _validate_factor(self, factor):
        """Check that a factor of a term is valid."""
        if len(factor) != 2:
            raise ValueError('Invalid factor {}.'.format(factor))

        index, action = factor

        if action not in self.actions:
            raise ValueError('Invalid action in factor {}. '
                             'Valid actions are: {}'.format(
                factor, self.actions))

        if not isinstance(index, int) or index < 0:
            raise ValueError('Invalid index in factor {}. '
                             'The index should be a non-negative '
                             'integer.'.format(factor))

    def _simplify(self, term, coefficient=1.0):
        """Simplifies a term."""
        if self.different_indices_commute:
            term = sorted(term, key=lambda factor: factor[0])
        return coefficient, tuple(term)

    def _parse_sequence(self, term):
        """Parse a term given as a sequence type (i.e., list, tuple, etc.).

        e.g. For QubitOperator:
            [('X', 2), ('Y', 0), ('Z', 3)] -> (('Y', 0), ('X', 2), ('Z', 3))
        """
        if not term:
            # Empty sequence
            return ()
        elif isinstance(term[0], int):
            # Single factor
            self._validate_factor(term)
            return (tuple(term),)
        else:
            # Check that all factors in the term are valid
            for factor in term:
                self._validate_factor(factor)

            # Return a tuple
            return tuple(term)

    def _parse_string(self, term):
        """Parse a term given as a string.

        e.g. For FermionOperator:
            "2^ 3" -> ((2, 1), (3, 0))
        """
        factors = term.split()

        # Convert the string representations of the factors to tuples
        processed_term = []
        for factor in factors:
            # Get the index and action string
            if self.action_before_index:
                # The index is at the end of the string; find where it starts.
                if not factor[-1].isdigit():
                    raise ValueError('Invalid factor {}.'.format(factor))
                index_start = len(factor) - 1

                while index_start > 0 and factor[index_start - 1].isdigit():
                    index_start -= 1
                if factor[index_start - 1] == '-':
                    raise ValueError('Invalid index in factor {}. '
                                     'The index should be a non-negative '
                                     'integer.'.format(factor))

                index = int(factor[index_start:])
                action_string = factor[:index_start]
            else:
                # The index is at the beginning of the string; find where
                # it ends
                if factor[0] == '-':
                    raise ValueError('Invalid index in factor {}. '
                                     'The index should be a non-negative '
                                     'integer.'.format(factor))
                if not factor[0].isdigit():
                    raise ValueError('Invalid factor {}.'.format(factor))
                index_end = 1
                while (index_end <= len(factor) - 1 and
                       factor[index_end].isdigit()):
                    index_end += 1

                index = int(factor[:index_end])
                action_string = factor[index_end:]

            # Convert the action string to an action
            if action_string in self.action_strings:
                action = self.actions[self.action_strings.index(action_string)]
            else:
                raise ValueError('Invalid action in factor {}. '
                                 'Valid actions are: {}'.format(
                    factor, self.action_strings))

            # Add the factor to the list as a tuple
            processed_term.append((index, action))

        # Return a tuple
        return tuple(processed_term)

    @property
    def constant(self):
        """The value of the constant term."""
        return self.terms.get((), 0.0)

    @constant.setter
    def constant(self, value):
        self.terms[()] = value

    @classmethod
    def zero(cls):
        """
        Returns:
            additive_identity (SymbolicOperator):
                A symbolic operator o with the property that o+x = x+o = x for
                all operators x of the same class.
        """
        return cls(term=None)

    @classmethod
    def identity(cls):
        """
        Returns:
            multiplicative_identity (SymbolicOperator):
                A symbolic operator u with the property that u*x = x*u = x for
                all operators x of the same class.
        """
        return cls(term=())

    def __str__(self):
        """Return an easy-to-read string representation."""
        if not self.terms:
            return '0'
        string_rep = '' if self.n_site == -1 else f'n_site: {self.n_site}\n'
        for term, coeff in sorted(self.terms.items()):
            if self._issmall(coeff):
                continue
            tmp_string = '{} ['.format(coeff)
            for factor in term:
                index, action = factor
                action_string = self.action_strings[self.actions.index(action)]
                if self.action_before_index:
                    tmp_string += '{}{} '.format(action_string, index)
                else:
                    tmp_string += '{}{} '.format(index, action_string)
            string_rep += '{}] +\n'.format(tmp_string.strip())
        return string_rep[:-3]

    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        return str(self)

    def __imul__(self, multiplier):
        """In-place multiply (*=) with scalar or operator of the same type.

        Default implementation is to multiply coefficients and
        concatenate terms.

        Args:
            multiplier(complex float, or SymbolicOperator): multiplier
        Returns:
            product (SymbolicOperator): Mutated self.
        """
        # Handle scalars.
        self.hash_cache = None
        if isinstance(multiplier, COEFFICIENT_TYPES):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle operator of the same type
        elif isinstance(multiplier, self.__class__):
            result_terms = dict()
            for left_term in self.terms:
                for right_term in multiplier.terms:
                    left_coefficient = self.terms[left_term]
                    right_coefficient = multiplier.terms[right_term]

                    new_coefficient = left_coefficient * right_coefficient
                    new_term = left_term + right_term

                    new_coefficient, new_term = self._simplify(
                        new_term, coefficient=new_coefficient)

                    # Update result dict.
                    if new_term in result_terms:
                        result_terms[new_term] += new_coefficient
                    else:
                        result_terms[new_term] = new_coefficient
            self.terms = result_terms
            return self

        # Invalid multiplier type
        else:
            raise TypeError('Cannot multiply {} with {}'.format(
                self.__class__.__name__, multiplier.__class__.__name__))

    def __mul__(self, multiplier):
        """Return self * multiplier for a scalar, or a SymbolicOperator.

        Args:
            multiplier: A scalar, or a SymbolicOperator.

        Returns:
            product (SymbolicOperator)

        Raises:
            TypeError: Invalid type cannot be multiply with SymbolicOperator.
        """
        if isinstance(multiplier, COEFFICIENT_TYPES + (type(self),)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError('Object of invalid type cannot multiply with ' +
                            type(self) + '.')

    def __iadd__(self, addend):
        """In-place method for += addition of SymbolicOperator.

        Args:
            addend (SymbolicOperator, or scalar): The operator to add.
                If scalar, adds to the constant term

        Returns:
            sum (SymbolicOperator): Mutated self.

        Raises:
            TypeError: Cannot add invalid type.
        """
        self.hash_cache = None
        if isinstance(addend, type(self)):
            for term in addend.terms:
                self.terms[term] = (self.terms.get(term, 0.0) +
                                    addend.terms[term])
                if self._issmall(self.terms[term]):
                    del self.terms[term]
        elif isinstance(addend, COEFFICIENT_TYPES):
            self.constant += addend
        else:
            raise TypeError('Cannot add invalid type to {}.'.format(type(self)))

        return self

    def __add__(self, addend):
        """
        Args:
            addend (SymbolicOperator): The operator to add.

        Returns:
            sum (SymbolicOperator)
        """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __radd__(self, addend):
        """
        Args:
            addend (SymbolicOperator): The operator to add.

        Returns:
            sum (SymbolicOperator)
        """
        return self + addend

    def __isub__(self, subtrahend):
        """In-place method for -= subtraction of SymbolicOperator.

        Args:
            subtrahend (A SymbolicOperator, or scalar): The operator to subtract
                if scalar, subtracts from the constant term.

        Returns:
            difference (SymbolicOperator): Mutated self.

        Raises:
            TypeError: Cannot subtract invalid type.
        """
        self.hash_cache = None
        if isinstance(subtrahend, type(self)):
            for term in subtrahend.terms:
                self.terms[term] = (self.terms.get(term, 0.0) -
                                    subtrahend.terms[term])
                if self._issmall(self.terms[term]):
                    del self.terms[term]
        elif isinstance(subtrahend, COEFFICIENT_TYPES):
            self.constant -= subtrahend
        else:
            raise TypeError('Cannot subtract invalid type from {}.'.format(
                type(self)))
        return self

    def __sub__(self, subtrahend):
        """
        Args:
            subtrahend (SymbolicOperator): The operator to subtract.

        Returns:
            difference (SymbolicOperator)
        """
        minuend = copy.deepcopy(self)
        minuend -= subtrahend
        return minuend

    def __rsub__(self, subtrahend):
        """
        Args:
            subtrahend (SymbolicOperator): The operator to subtract.

        Returns:
            difference (SymbolicOperator)
        """
        return -1 * self + subtrahend

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  SymbolicOperator and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of SymbolicOperator.

        Raises:
          TypeError: Object of invalid type cannot multiply SymbolicOperator.
        """
        if not isinstance(multiplier, COEFFICIENT_TYPES):
            raise TypeError('Object of invalid type cannot multiply with ' +
                            type(self) + '.')
        return self * multiplier

    def __truediv__(self, divisor):
        """
        Return self / divisor for a scalar.

        Note:
            This is always floating point division.

        Args:
          divisor: A scalar to divide by.

        Returns:
          A new instance of SymbolicOperator.

        Raises:
          TypeError: Cannot divide local operator by non-scalar type.

        """
        if not isinstance(divisor, COEFFICIENT_TYPES):
            raise TypeError('Cannot divide ' + type(self) +
                            ' by non-scalar type.')
        return self * (1.0 / divisor)

    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)

    def __itruediv__(self, divisor):
        if not isinstance(divisor, COEFFICIENT_TYPES):
            raise TypeError('Cannot divide ' + type(self) +
                            ' by non-scalar type.')
        self *= (1.0 / divisor)
        return self

    def __idiv__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__itruediv__(divisor)

    def __neg__(self):
        """
        Returns:
            negation (SymbolicOperator)
        """
        return -1 * self

    def __pow__(self, exponent):
        """Exponentiate the SymbolicOperator.

        Args:
            exponent (int): The exponent with which to raise the operator.

        Returns:
            exponentiated (SymbolicOperator)

        Raises:
            ValueError: Can only raise SymbolicOperator to non-negative
                integer powers.
        """
        # Handle invalid exponents.
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError(
                'exponent must be a non-negative int, but was {} {}'.format(
                    type(exponent), repr(exponent)))

        # Initialized identity.
        exponentiated = self.__class__(())

        # Handle non-zero exponents.
        for _ in range(exponent):
            exponentiated *= self
        return exponentiated

    def __eq__(self, other):
        """Approximate numerical equality (not true equality)."""
        return self.isclose(other)

    def __ne__(self, other):
        return not (self == other)

    def __iter__(self):
        self._iter = iter(self.terms.items())
        return self

    def __next__(self):
        term, coeff = next(self._iter)
        return term, coeff

    def isclose(self, other, tol=EQ_TOLERANCE):
        """Check if other (SymbolicOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each term in self and other is
        less than EQ_TOLERANCE

        Args:
            other(SymbolicOperator): SymbolicOperator to compare against.
        """
        if not isinstance(self, type(other)):
            return NotImplemented

        # terms which are in both:
        for term in set(self.terms).intersection(set(other.terms)):
            a = self.terms[term]
            b = other.terms[term]
            if not (isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr)):
                tol *= max(1, abs(a), abs(b))
            if self._issmall(a - b, tol) is False:
                return False
        # terms only in one (compare to 0.0 so only abs_tol)
        for term in set(self.terms).symmetric_difference(set(other.terms)):
            if term in self.terms:
                if self._issmall(self.terms[term], tol) is False:
                    return False
            else:
                if self._issmall(other.terms[term], tol) is False:
                    return False
        return True

    def compress(self, abs_tol=EQ_TOLERANCE):
        """
        Eliminates all terms with coefficients close to zero and removes
        small imaginary and real parts.

        Args:
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        new_terms = {}
        for term in self.terms:
            coeff = self.terms[term]

            if isinstance(coeff, sympy.Expr):
                if sympy.simplify(sympy.im(coeff) <= abs_tol) == True:
                    coeff = sympy.re(coeff)
                if sympy.simplify(sympy.re(coeff) <= abs_tol) == True:
                    coeff = 1j * sympy.im(coeff)
                if (sympy.simplify(abs(coeff) <= abs_tol) != True):
                    new_terms[term] = coeff
                continue

            # Remove small imaginary and real parts
            if abs(coeff.imag) <= abs_tol:
                coeff = coeff.real
            if abs(coeff.real) <= abs_tol:
                coeff = 1.j * coeff.imag

            # Add the term if the coefficient is large enough
            if abs(coeff) > abs_tol:
                new_terms[term] = coeff

        self.terms = new_terms

    def induced_norm(self, order=1):
        r"""
        Compute the induced p-norm of the operator.

        If we represent an operator as
        $\sum_{j} w_j H_j$
        where $w_j$ are scalar coefficients then this norm is
        $\left(\sum_{j} \| w_j \|^p \right)^{\frac{1}{p}}$
        where $p$ is the order of the induced norm

        Args:
            order(int): the order of the induced norm.
        """
        norm = 0.
        for coefficient in self.terms.values():
            norm += abs(coefficient) ** order
        return norm ** (1. / order)

    def many_body_order(self):
        """Compute the many-body order of a SymbolicOperator.

        The many-body order of a SymbolicOperator is the maximum length of
        a term with nonzero coefficient.

        Returns:
            int
        """
        if not self.terms:
            # Zero operator
            return 0
        else:
            return max(
                len(term)
                for term, coeff in self.terms.items()
                if (self._issmall(coeff) is False))

    @classmethod
    def accumulate(cls, operators, start=None):
        """Sums over SymbolicOperators."""
        total = copy.deepcopy(start or cls.zero())
        for operator in operators:
            total += operator
        return total

    def get_operators(self):
        """Gets a list of operators with a single term.

        Returns:
            operators([self.__class__]): A generator of the operators in self.
        """
        for term, coefficient in self.terms.items():
            yield self.__class__(term, coefficient)

    def get_operator_groups(self, num_groups):
        """Gets a list of operators with a few terms.
        Args:
            num_groups(int): How many operators to get in the end.

        Returns:
            operators([self.__class__]): A list of operators summing up to
                self.
        """
        if num_groups < 1:
            warnings.warn('Invalid num_groups {} < 1.'.format(num_groups),
                          RuntimeWarning)
            num_groups = 1

        operators = self.get_operators()
        num_groups = min(num_groups, len(self.terms))
        for i in range(num_groups):
            yield self.accumulate(
                itertools.islice(operators,
                                 len(range(i, len(self.terms), num_groups))))


#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""QubitOperator stores a sum of Pauli operators acting on qubits."""
from typing import Tuple

import numpy as np

from copy import deepcopy

# Define products of all Pauli operators for symbolic multiplication.
_PAULI_OPERATOR_PRODUCTS = {
    ('I', 'I'): (1., 'I'),
    ('I', 'X'): (1., 'X'),
    ('X', 'I'): (1., 'X'),
    ('I', 'Y'): (1., 'Y'),
    ('Y', 'I'): (1., 'Y'),
    ('I', 'Z'): (1., 'Z'),
    ('Z', 'I'): (1., 'Z'),
    ('X', 'X'): (1., 'I'),
    ('Y', 'Y'): (1., 'I'),
    ('Z', 'Z'): (1., 'I'),
    ('X', 'Y'): (1.j, 'Z'),
    ('X', 'Z'): (-1.j, 'Y'),
    ('Y', 'X'): (-1.j, 'Z'),
    ('Y', 'Z'): (1.j, 'X'),
    ('Z', 'X'): (1.j, 'Y'),
    ('Z', 'Y'): (-1.j, 'X')
}

##### mizore modification
_pauli_name_map = {
    "I": 0,
    "X": 1,
    "Y": 2,
    "Z": 3
}

_pauli_index_map = ["I", "X", "Y", "Z"]

PauliTuple = Tuple[Tuple[int, str], ...]


#####

class QubitOperator(SymbolicOperator):
    """
    A sum of terms acting on qubits, e.g., 0.5 * 'X0 X5' + 0.3 * 'Z1 Z2'.

    A term is an operator acting on n qubits and can be represented as:

    coefficient * local_operator[0] x ... x local_operator[n-1]

    where x is the tensor product. A local operator is a Pauli operator
    ('I', 'X', 'Y', or 'Z') which acts on one qubit. In math notation a term
    is, for example, 0.5 * 'X0 X5', which means that a Pauli X operator acts
    on qubit 0 and 5, while the identity operator acts on all other qubits.

    A QubitOperator represents a sum of terms acting on qubits and overloads
    operations for easy manipulation of these objects by the user.

    Note for a QubitOperator to be a Hamiltonian which is a hermitian
    operator, the coefficients of all terms must be real.

    .. code-block:: python

        hamiltonian = 0.5 * QubitOperator('X0 X5') + 0.3 * QubitOperator('Z0')

    QubitOperator is a subclass of SymbolicOperator. Importantly, it has
    attributes set as follows::

        actions = ('X', 'Y', 'Z')
        action_strings = ('X', 'Y', 'Z')
        action_before_index = True
        different_indices_commute = True

    See the documentation of SymbolicOperator for more details.

    Example:
        .. code-block:: python

            ham = ((QubitOperator('X0 Y3', 0.5)
                    + 0.6 * QubitOperator('X0 Y3')))
            # Equivalently
            ham2 = QubitOperator('X0 Y3', 0.5)
            ham2 += 0.6 * QubitOperator('X0 Y3')

    Note:
        Adding QubitOperators is faster using += (as this
        is done by in-place addition). Specifying the coefficient
        during initialization is faster than multiplying a QubitOperator
        with a scalar.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def actions(self):
        """The allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return True

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return True

    ##### mizore modification
    @property
    def n_qubit(self):
        return self.n_site

    @n_qubit.setter
    def n_qubit(self, value):
        self.n_site = value

    def qset_op_weight_omit_const(self):
        for op_tuple, weight in self.terms.items():
            if len(op_tuple) == 0:
                continue
            qset = [i for i, _ in op_tuple]
            ops = [_pauli_name_map[op] for _, op in op_tuple]
            yield qset, ops, weight

    def qset_op_weight(self):
        for op_tuple, weight in self.terms.items():
            qset = [i for i, _ in op_tuple]
            ops = [_pauli_name_map[op] for _, op in op_tuple]
            yield qset, ops, weight

    def get_qset(self):
        qset_dict = {}
        for ops_tuple, weight in self.terms.items():
            for i, _ in ops_tuple:
                qset_dict[i] = True
        qset = list(qset_dict)
        qset.sort()
        return qset

    @classmethod
    def from_qset_op(cls, qset, pauli_op):
        op_tuple = tuple([(qset[i], _pauli_index_map[pauli_op[i]]) for i in range(len(qset))])
        op = QubitOperator()
        op.terms[op_tuple] = 1.0
        return op

    @classmethod
    def from_qset_op_weight(cls, qset, pauli_op, weight):
        return weight * QubitOperator.from_qset_op(qset, pauli_op)

    @classmethod
    def from_pauli_tuple(cls, pauli_tuple: PauliTuple):
        op = QubitOperator()
        op.terms[pauli_tuple] = 1.0
        return op

    @classmethod
    def from_pauli_tensor(cls, coeffs, pauli_tensor):
        terms = defaultdict(lambda: 0.0)
        if len(coeffs) == 0:
            return QubitOperator()
        for i in range(len(pauli_tensor)):
            pword_tensor = pauli_tensor[i]
            coeff = coeffs[i]
            pauli_tuple = tuple((i, _pauli_index_map[int(p)]) for i, p in enumerate(pword_tensor) if p != 0)
            terms[pauli_tuple] += coeff
        op = QubitOperator.from_terms_dict(dict(terms))
        op.n_qubit = len(pword_tensor)
        return op

    def get_unique_op_tuple(self) -> Tuple[Tuple, complex]:
        op_tuples = list(self.terms.items())
        if len(op_tuples) != 1:
            raise Exception("There are more than one terms in" + str(op_tuples))
        return op_tuples[0]

    def replica(self):
        return deepcopy(self)

    def remove_constant(self) -> Tuple[QubitOperator, complex]:
        """
        Remove the constant part of the operator

        Returns:
            (new operator, constant)
        """
        new_op = self.replica()
        if () in new_op.terms.keys():
            const = new_op.terms[()]
            del new_op.terms[()]
        else:
            const = 0.0
        return new_op, const

    def get_l1_norm_omit_const(self):
        l1_norm = 0.0
        for weight in self.terms.values():
            l1_norm += abs(weight)
        l1_norm -= self.terms.get((), 0.0)
        return l1_norm

    def iter_sub_ops(self):
        for key, value in self.terms.items():
            if key == ():
                pass
            new_op = QubitOperator()
            new_op.terms = {key: 1.0}
            yield new_op, value

    def count_n_qubit(self):
        return max(self.get_qset()) + 1

    #####

    def renormalize(self):
        """Fix the trace norm of an operator to 1"""
        norm = self.induced_norm(2)
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError('Cannot renormalize empty or zero operator')
        else:
            self /= norm

    def _simplify(self, term, coefficient=1.0):
        """Simplify a term using commutator and anti-commutator relations."""
        if not term:
            return coefficient, term

        term = sorted(term, key=lambda factor: factor[0])

        new_term = []
        left_factor = term[0]
        for right_factor in term[1:]:
            left_index, left_action = left_factor
            right_index, right_action = right_factor

            # Still on the same qubit, keep simplifying.
            if left_index == right_index:
                new_coefficient, new_action = _PAULI_OPERATOR_PRODUCTS[
                    left_action, right_action]
                left_factor = (left_index, new_action)
                coefficient *= new_coefficient

            # Reached different qubit, save result and re-initialize.
            else:
                if left_action != 'I':
                    new_term.append(left_factor)
                left_factor = right_factor

        # Save result of final iteration.
        if left_factor[1] != 'I':
            new_term.append(left_factor)

        return coefficient, tuple(new_term)

    # Add tensor utilities
    def get_one_hot_tensor(self):
        n_qubit = self.n_qubit if self.n_qubit > 0 else self.count_n_qubit()
        coeffs = []
        pwords = []
        for pword, coeff in self:
            pwords.append(get_pword_one_hot_tensor(pword, n_qubit))
            coeffs.append(coeff)
        return np.array(coeffs), np.stack(pwords)

    def get_pauli_tensor(self):
        n_qubit = self.n_qubit if self.n_qubit > 0 else self.count_n_qubit()
        coeffs = []
        pwords = []
        for pword, coeff in self:
            pwords.append(get_pword_tensor(pword, n_qubit))
            coeffs.append(coeff)
        return np.array(coeffs), np.stack(pwords)


_pauli_to_index = {"X": 0, "Y": 1, "Z": 2}


def get_pword_one_hot_tensor(pword, n_qubit):
    pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit][_pauli_to_index[pauli]] = 1.0
    return np.array(pauli_tensor)


def get_pword_tensor(pword, n_qubit):
    pauli_tensor = [0 for _ in range(n_qubit)]
    for i_qubit, pauli in pword:
        pauli_tensor[i_qubit] = _pauli_name_map[pauli]
    return np.array(pauli_tensor)
