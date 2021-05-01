# -*- coding: utf-8 -*-

"""Provide road link classes for the OpenDRIVE implementation."""


class RoadLink:
    """ Represents the links of the Road """

    def __init__(self, link_id=None, predecessor=None, successor=None, neighbors=None):
        self.id = link_id
        self._predecessor = predecessor
        self._successor = successor
        self._neighbors = [] if neighbors is None else neighbors

    def __str__(self):
        return " > link id " + str(self._id) + " | successor: " + str(self._successor)

    @property
    def id(self) -> int:
        """ ID of the Road link"""
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = int(value) if value is not None else None

    @property
    def predecessor(self):
        """ Predecessor Road or Junction """
        return self._predecessor

    @predecessor.setter
    def predecessor(self, value):
        if not isinstance(value, Predecessor) and value is not None:
            raise TypeError("Value must be Predecessor")
        self._predecessor = value

    @property
    def successor(self):
        """ Successor Road or Junction """
        return self._successor

    @successor.setter
    def successor(self, value):
        if not isinstance(value, Successor) and value is not None:
            raise TypeError("Value must be Successor")
        self._successor = value

    @property
    def neighbors(self):
        """ Neighbouring Roads """
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Neighbor) for x in value):
            raise TypeError("Value must be list of instances of Neighbor.")
        self._neighbors = value

    def add_neighbor(self, value):
        """ Add a neighbouring Road to the list """
        if not isinstance(value, Neighbor):
            raise TypeError("Value must be Neighbor")
        self._neighbors.append(value)


class Predecessor:
    """ Represents a predecessor Road or Junction in the OpenDrive standard """

    def __init__(self, element_type=None, element_id=None, contact_point=None):
        self._element_type = element_type
        self._element_id = element_id
        self._contact_point = contact_point

        self._element = None

    def __str__(self):
        return (
            str(self._element_type)
            + " with id "
            + str(self._element_id)
            + " contact at "
            + str(self._contact_point)
        )

    @property
    def element_type(self) -> str:
        """ Whether the link is a Road or a Junction """
        return self._element_type

    @element_type.setter
    def element_type(self, value: str):
        if value not in ["road", "junction"]:
            raise AttributeError("Value must be road or junction")
        self._element_type = value

    @property
    def element_id(self) -> int:
        """ The ID of the connecting element """
        return self._element_id

    @element_id.setter
    def element_id(self, value: int):
        self._element_id = int(value)

    @property
    def element(self):
        """ The connecting element """
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def contact_point(self) -> str:
        """ Contact point of the connecting element. Either 'start' or 'end' """
        return self._contact_point

    @contact_point.setter
    def contact_point(self, value: str):
        if value not in ["start", "end"] and value is not None:
            raise AttributeError("Value must be start or end or None")
        self._contact_point = value


class Successor(Predecessor):
    """ Represents a Successor link in the OpenDrive standard """


class Neighbor:
    """ Represents a Neighbouring Road in the OpenDrive standard """

    def __init__(self, side=None, element_id=None, direction=None):
        self._side = side
        self._element_id = element_id
        self._direction = direction
        self._element = None

    @property
    def side(self) -> str:
        """ Side of relative to the direction of the current Road. Either 'left' or 'right' """
        return self._side

    @side.setter
    def side(self, value: str):
        if value not in ["left", "right"]:
            raise AttributeError("Value must be left or right")
        self._side = value

    @property
    def element_id(self) -> int:
        """ ID of the neighbouring Road """
        return self._element_id

    @element_id.setter
    def element_id(self, value: int):
        self._element_id = int(value)

    @property
    def element(self):
        """ The connecting element """
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def direction(self) -> str:
        """ Direction of the neighbouring Road relative to the direction of the current Road.
        Either 'same' or 'opposite'
        """
        return self._direction

    @direction.setter
    def direction(self, value: str):
        if value not in ["same", "opposite"]:
            raise AttributeError("Value must be same or opposite")
        self._direction = value
