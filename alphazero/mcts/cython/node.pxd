# cython: language_level=3
"""Cython declaration file for node.pyx"""

import numpy as np
cimport numpy as np

ctypedef np.float32_t FLOAT_t

cdef class CythonMCTSNode:
    cdef public int visit_count
    cdef public double value_sum
    cdef public double prior
    cdef public bint _is_expanded
    cdef public bint _is_terminal
    cdef public double _terminal_value
    cdef public dict _children
    cdef public object _legal_actions

    cpdef bint is_expanded(self)
    cpdef bint is_terminal(self)
    cpdef void set_terminal(self, double value)
    cpdef double get_terminal_value(self)
    cpdef void expand(self, np.ndarray[FLOAT_t, ndim=1] priors,
                      np.ndarray[FLOAT_t, ndim=1] legal_mask)
    cpdef tuple select_child(self, double c_puct)
    cpdef dict get_children(self)
    cpdef object get_child(self, int action)
    cpdef void update(self, double value)
    cpdef np.ndarray get_visit_counts(self, int num_actions)
    cpdef np.ndarray get_policy(self, int num_actions, double temperature=*)
