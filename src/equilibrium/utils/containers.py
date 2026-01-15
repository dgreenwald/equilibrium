#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:22:02 2025

@author: dan
"""

import logging
from collections import OrderedDict
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class MyOrderedDict(OrderedDict):
    """OrderedDict plus addition operator"""

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

    def __add__(self, other):
        temp = self.copy()
        temp.update(OrderedDict(other))
        return MyOrderedDict(temp)

    def __radd__(self, other):
        temp = OrderedDict(other)
        temp.update(self)
        return MyOrderedDict(temp)


class PresetDict(dict):
    """Preset dict that will not update if key already set"""

    def __init__(self, other=None, verbose=False, **kwargs):
        super().__init__()
        self.verbose = verbose
        self.update(other, **kwargs)

    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, value)
        elif self.verbose:
            logger.info("PresetDict: ignoring key '%s'", key)

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def overwrite_item(self, key, value):

        super().__setitem__(key, value)

    def overwrite_update(self, other):

        super().update(other)
