# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict
from itertools import chain

class Groups(object):
    """Helper class to manage groups with following abilities:
        - keeps list of groups and their elements (each element may by only in one group)
        - may determine the group of an item
        - may return a list with items of a single group or all the items from all the groups

        The class uses OrderedDict for storage and thus keeps the order.
    """
    def __init__(self, groups):
        """groups - dict like object with key : [item1, item2, ...] pairs"""
        self.groups = OrderedDict(groups)
        self.match = OrderedDict()
        for groups, keylist in self.groups.items():
            for key in keylist:
                self.match[key] = groups

    def __repr__(self):
        return self.groups.__repr__()

    def keys(self, group=None):
        """Returns iterable with items of a group or all the items of all the groups"""
        return self.groups[group] if group else chain(*self.groups.values())

    def items(self):
        """Unpack each group and iterate group, key pairs"""
        for group, keys in self.groups.items():
            for key in keys:
                yield group, key

    def __contains__(self, item):
        """Checks that item is known"""
        return item in self.match

    def __getitem__(self, item):
        """Returns a group of an item"""
        return self.match[item]

    def hasgroup(self, group):
        return group in self.groups

    def containsall(self, items):
        for item in items:
            if not item in self:
                return False
        return True

    def samegroup(self, items):
        return len({self.match[item] for item in items})==1

    def group(self, item):
        return self.groups[self[item]]

class Categories(object):
    """A set of Groups instances"""
    def __init__(self, groups, recursive=False):
        self.recursive = recursive
        self.__categories__ = OrderedDict([ (k, Groups(g)) for k, g in groups.items() ])

    def __contains__(self, item, recursive=None):
        """Checks if the item belongs at least to the one of groupings"""
        for group in self.__categories__.values():
            if item in group:
                return True

            if self.recursive and item in group.groups:
                return True

        return False

    def get_group(self, item, cat):
        return self.__categories__[cat].group(item)

    def group(self, item, cat=None):
        if cat:
            return self.__categories__[cat][item]

        for group, items in self.__categories__.items():
            if item in items:
                return items[item]
        return None

    def groups(self, item):
        res = [ group[item] for cat, group in self.__categories__.items() if item in group ]
        if self.recursive:
            for cat, group in self.__categories__.items():
                if not group.hasgroup(item):
                    continue
                groupitems = group.keys(item)
                for newcat, newgroup in self.__categories__.items():
                    if not newgroup.samegroup( groupitems ):
                        continue
                    res.append( newgroup[groupitems[0]] )
            return list(OrderedDict.fromkeys(res))
        return res

    def categories(self, item):
        res = [ cat for cat, group in self.__categories__.items() if item in group ]
        if self.recursive:
            for cat, group in self.__categories__.items():
                if not group.hasgroup(item):
                    continue
                groupitems = group.keys(item)
                for newcat, newgroup in self.__categories__.items():
                    if not newgroup.samegroup( groupitems ):
                        continue
                    res.append( newcat )
            return list(OrderedDict.fromkeys(res))
        return res

    def items(self, item):
        return self.itemdict(item).items()

    def itemdict(self, item):
        res = OrderedDict([ (cat, group[item]) for cat, group in self.__categories__.items() if item in group ])
        if self.recursive:
            for cat, group in self.__categories__.items():
                if not group.hasgroup(item):
                    continue
                groupitems = group.keys(item)
                for newcat, newgroup in self.__categories__.items():
                    if not newgroup.samegroup( groupitems ):
                        continue
                    res[newcat]=newgroup[groupitems[0]]
        return res

    def format(self, item, fmt, **kwargs):
        if isinstance(fmt, basestring):
            kwargs = OrderedDict([('self', item)]+list(self.itemdict(item).items())+list(kwargs.items()))
            return fmt.format(**kwargs)

        return type(fmt)(self.format(item, s, **kwargs) for s in fmt)

    def format_splitjoin(self, item, fmt, sep='.', filter=(''), prepend='', **kwargs):
        return sep.join(s for s in self.format(item, prepend.split(sep)+fmt.split(sep), **kwargs) if not s in filter)

class GroupedDict(OrderedDict):
    """OrderedDict implementation with:
        - if key is present the behaviour is regular
        - if key is missing, checks if key belongs to a group and uses group name instead of key"""
    def __init__(self, groups, *args, **kwargs):
        """Obligatory argument: groups - Groups object or dict like object with key : [item1, item2, ...] pairs"""
        self.groups = groups
        if not isinstance(self.groups, Groups):
            self.groups = Groups(self.groups)

        super(GroupedDict, self).__init__(*args, **kwargs)

    def __contains__(self, key):
        """Checks if key or its group present in dictionary"""
        contains = super(OrderedDict, self).__contains__
        return contains(key) or key in self.groups and contains(self.groups.group(key))

    def __missing__(self, key):
        """Return value by group name instead of a key name"""
        if key in self.groups:
            return self[self.groups[key]]

        raise KeyError(key)

    def subkeys(self, group=None):
        """The same as key but replace each group by its elements"""
        return self.groups.keys( group )

    def subitems(self):
        """The same as items but replace each group by its elements"""
        for group, key in self.groups.items():
            yield key, self[group]

    def subvalues(self):
        """The same as items but replace each group by its elements"""
        for group, key in self.groups.items():
            yield self[group]

class CatDict(OrderedDict):
    """OrderedDict implementation with:
        - if key is present the behaviour is regular
        - if key is missing, checks if key belongs to a group and uses group name instead of key"""
    def __init__(self, categories, *args, **kwargs):
        """Obligatory argument: groups - Groups object or dict like object with key : [item1, item2, ...] pairs"""
        self.categories = categories
        if not isinstance(self.categories, Categories):
            self.categories = Categories(self.groups)

        super(CatDict, self).__init__(*args, **kwargs)

    def __contains__(self, key):
        """Checks if key or its group present in dictionary"""
        contains = super(CatDict, self).__contains__
        if contains(key):
            return True

        for group in self.categories.groups(key):
            if contains(group):
                return True

        return False

    def __missing__(self, key):
        """Return value by group name instead of a key name"""
        for group in self.categories.groups(key):
            if super(CatDict, self).__contains__(group):
                return self[group]

        raise KeyError(key)

