# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict

class Groups(object):
    def __init__(self, groups):
        self.groups = groups
        self.match = OrderedDict()
        for groups, keylist in self.groups.items():
            for key in keylist:
                self.match[key] = groups

    def __repr__(self):
        return self.groups.__repr__()

    def group(self, item):
        return self.match[item]

    def items(self, group):
        return self.groups[group]

    def __contains__(self, item):
        return item in self.match

class GroupedDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        self.groups = Groups(kwargs.pop( 'groups' ))

        super(GroupedDict, self).__init__(*args, **kwargs)

    def __contains__(self, key):
        contains = super(OrderedDict, self).__contains__
        return contains(key) or key in self.groups and contains(self.groups.group(key))

    def __getitem__(self, key, default)





