# -*- coding: utf-8 -*-

from typing import List

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from igp2.opendrive.elements.road_lanes import Lane


class JunctionPriority:
    """ Priority specification between incoming and connecting roads. """
    def __init__(self, high_id: int, low_id: int):
        self.high_id = high_id
        self.low_id = low_id
        self.high = None
        self.low = None

    def __repr__(self):
        return f"{self.high_id} > {self.low_id}"


class JunctionLaneLink:
    """ Lane connections between the incoming road and the connecting road """

    def __init__(self):
        self._from_id = None
        self._from_lane = None
        self._to_id = None
        self._to_lane = None

    def __str__(self):
        return str(self._from_id) + " > " + str(self._to_id)

    @property
    def from_id(self) -> int:
        """ ID of lane on the incoming road """
        return self._from_id

    @from_id.setter
    def from_id(self, value: int):
        self._from_id = int(value)

    # @property
    # def from_lane(self) -> "Lane":
    #     """ ID of lane on the incoming road """
    #     return self._from_lane
    #
    # @from_lane.setter
    # def from_lane(self, value: "Lane"):
    #     self._from_lane = value

    @property
    def to_id(self):
        """ ID of lane on the connecting road """
        return self._to_id

    @to_id.setter
    def to_id(self, value: int):
        self._to_id = int(value)

    @property
    def to_lane(self) -> "Lane":
        """ ID of lane on the incoming road """
        return self._to_lane

    @to_lane.setter
    def to_lane(self, value: "Lane"):
        self._to_lane = value


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
    def lane_links(self) -> List[JunctionLaneLink]:
        """ List of LaneLinks between lanes of the incoming and connecting road """
        return self._lane_links

    def add_lane_link(self, lane_link: JunctionLaneLink):
        """ Add a new LaneLink to the Junction

        Args:
          lane_link: The LaneLink object to add
        """
        if not isinstance(lane_link, JunctionLaneLink):
            raise TypeError("Has to be of instance LaneLink")

        self._lane_links.append(lane_link)


class Junction:
    """ Represents a Junction object in the OpenDrive standard"""

    # TODO controller

    def __init__(self):
        self._id = None
        self._name = None
        self._boundary = None
        self._connections = []
        self._priorities = []
        self._junction_group = None
        self._roads = []

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

    @property
    def priorities(self) -> List[JunctionPriority]:
        return self._priorities

    @property
    def junction_group(self) -> "JunctionGroup":
        return self._junction_group

    @junction_group.setter
    def junction_group(self, junction_group: "JunctionGroup"):
        self._junction_group = junction_group

    @property
    def in_roundabout(self) -> bool:
        """ True if the junction is a roudabout junction. """
        return self.junction_group is not None and self.junction_group.type == "roundabout"

    @property
    def roads(self) -> List["Road"]:
        if not self._roads:
            self._roads = self.get_all_roads()
        return self._roads

    def add_connection(self, connection: Connection):
        """ Add a new connection to the Junction

        Args:
            connection: The Connection object to add
        """
        if not isinstance(connection, Connection):
            raise TypeError("Has to be of instance Connection")
        self._connections.append(connection)

    def add_priority(self, priority: JunctionPriority):
        """ Add a new priority field to the Junction

        Args:
            priority: The JunctionPriority object to add
        """
        if not isinstance(priority, JunctionPriority):
            raise TypeError("Must be of instance JunctionPriority")
        self._priorities.append(priority)

    def get_all_roads(self) -> List["Road"]:
        """ Return all roads that are part of this Junction.
        Warning: This function assumes that all roads in the junction are connecting roads.
        """
        ret = []
        for connection in self._connections:
            if connection.connecting_road not in ret:
                ret.append(connection.connecting_road)
        return ret

    def get_all_connecting_roads(self, incoming_road: "Road") -> List["Road"]:
        """ Return all connecting roads of the given incoming Road.

        Args:
            incoming_road: The incoming Road object

        Returns:
            List of all connecting roads
        """
        ret = []
        for connection in self._connections:
            if connection.incoming_road == incoming_road:
                ret.append(connection.connecting_road)
        return ret

    def get_all_connecting_lanes(self, incoming_lane: Lane) -> List[Lane]:
        """ Return all connecting lanes of the given incoming Lane.

        Args:
            incoming_lane: The incoming Lane object

        Returns:
            List of connecting Lanes
        """
        ret = []
        for connection in self._connections:
            if connection.incoming_road == incoming_lane.parent_road:
                for lane_link in connection.lane_links:
                    if lane_link.from_id == incoming_lane.id:
                        ret.append(lane_link.to_lane)
        return ret

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
        if boundary.geom_type == "Polygon":
            self._boundary = Polygon(boundary.exterior)
        else:
            self._boundary = MultiPolygon([Polygon(polygon.exterior) for polygon in boundary.geoms])


class JunctionGroup:
    """ A Junction group.

    Reference: OpenDrive 1.6.1 - Section 10.5
    """
    def __init__(self, name: str, group_id: int, group_type: str):
        self.name = name
        self.id = group_id

        if group_type not in ["roundabout", "unknown"]:
            raise ValueError("Junction group type must be roundabout or unknown")
        self.type = group_type

        self.junctions = []

    def add_junction(self, junction: Junction):
        """ Add a new Junction element to the group """
        if not isinstance(junction, Junction):
            raise ValueError("Given object is not a Junction")
        self.junctions.append(junction)

