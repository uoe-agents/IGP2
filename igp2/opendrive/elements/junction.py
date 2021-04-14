# -*- coding: utf-8 -*-

from typing import List

from shapely.geometry import Polygon
from shapely.ops import unary_union


class LaneLink:
    """ """

    def __init__(self):
        self._from = None
        self._to = None

    def __str__(self):
        return str(self._from) + " > " + str(self._to)

    @property
    def from_id(self) -> int:
        """ ID of lane on the incoming road """
        return self._from

    @from_id.setter
    def from_id(self, value: int):
        self._from = int(value)

    @property
    def to_id(self):
        """ ID of lane on the connecting road """
        return self._to

    @to_id.setter
    def to_id(self, value: int):
        self._to = int(value)


class Connection:
    """ Object representing a Junction in the OpenDrive standard """

    def __init__(self):
        self._id = None
        self._incoming_road = None
        self._connecting_road = None
        self._contact_point = None
        self._lane_links = []

    def __repr__(self):
        return f"{self._incoming_road.id} > {self._connecting_road.id}"

    @property
    def id(self) -> int:
        """ Unique ID for the Connection """
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = int(value)

    @property
    def incoming_road(self):
        """ The incoming Road object"""
        return self._incoming_road

    @incoming_road.setter
    def incoming_road(self, value):
        self._incoming_road = value

    @property
    def connecting_road(self):
        """ The connecting Road object """
        return self._connecting_road

    @connecting_road.setter
    def connecting_road(self, value: int):
        self._connecting_road = value

    @property
    def contact_point(self) -> str:
        """ Contact point of the connecting road. Either 'start' or 'end' """
        return self._contact_point

    @contact_point.setter
    def contact_point(self, value: str):
        if value not in ["start", "end"]:
            raise AttributeError("Contact point can only be start or end.")

        self._contact_point = value

    @property
    def lane_links(self) -> List[LaneLink]:
        """ List of LaneLinks between lanes of the incoming and connecting road """
        return self._lane_links

    def add_lane_link(self, lane_link: LaneLink):
        """ Add a new LaneLink to the Junction

        Args:
          lane_link: The LaneLink object to add
        """
        if not isinstance(lane_link, LaneLink):
            raise TypeError("Has to be of instance LaneLink")

        self._lane_links.append(lane_link)


class Junction:
    """ Represents a Junction object in the OpenDrive standard"""

    # TODO priority
    # TODO controller

    def __init__(self):
        self._id = None
        self._name = None
        self._boundary = None
        self._connections = []

    @property
    def id(self) -> int:
        """ Unique ID of the junction """
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = int(value)

    @property
    def name(self) -> str:
        """ Name for the junction """
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = str(value)

    @property
    def connections(self) -> List[Connection]:
        """ The list of connections in the junction"""
        return self._connections

    def add_connection(self, connection: Connection):
        """ Add a New connection to the Junction

        Args:
            connection: The Connection object to add
        """
        if not isinstance(connection, Connection):
            raise TypeError("Has to be of instance Connection")
        self._connections.append(connection)

    @property
    def boundary(self):
        """ The boundary of the Junction formed as the union of all roads in the Junction"""
        return self._boundary

    def calculate_boundary(self):
        """ Calculate the boundary of the Junction given the list of roads in it """
        def extend_boundary(_road):
            if _road.id not in visited and _road.junction is not None and \
                    _road.junction.id == self.id:
                visited.add(_road.id)
                return unary_union([boundary, _road.boundary])
            return boundary

        boundary = Polygon()
        visited = set()
        for connection in self._connections:
            boundary = extend_boundary(connection.incoming_road)
            boundary = extend_boundary(connection.connecting_road)
        self._boundary = Polygon(boundary.exterior)
