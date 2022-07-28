#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.mpc.data.dynamic_data_base import (
    _DynamicDataBase,
)
from pyomo.contrib.mpc.data.get_cuid import (
    get_time_indexed_cuid,
)


class ScalarData(_DynamicDataBase):
    """
    An object to store scalar data associated with time-indexed
    variables.
    """

    def __init__(self, data, time_set=None):
        """
        Arguments:
        ----------
        data: dict or ComponentMap
            Maps variables, names, or CUIDs to lists of values
        """
        for key, val in data.items():
            # Is there a better way to check if val is iterable?
            if hasattr(val, "__iter__") or hasattr(val, "__getitem__"):
                raise TypeError(
                    "Value %s corresponding to key %s is not a scalar"
                    % (val, key)
                )
        super().__init__(data, time_set=time_set)

    def to_serializable(self):
        """
        Convert to json-serializable object.

        """
        data = {str(cuid): val for cuid, val in self._data.items()}
        return data
