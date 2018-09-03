# -*- coding: UTF-8 -*-
# The Robot class and its dependencies, which describe the Osiris robot and its surroundings
# Magnus Lindhé, 2018

class World:
    """Defines the obstacles and keeps track of which obstacles have been cleaned by the side brush."""

    def __init__(self, worldFileName):
        """Read world information from file "worldFileName"."""

        # Bitmap matrix of obstacles, each element represents a quadratic grid.
        # Values: 0 - empty, 1 - occupied and not cleaned, 2 - occupied and cleaned
        self.obstacles = 0

        # Coordinates for the lower left corner of the bitmap (m,m)
        self.origin = 0

        # Side length of the bitmap (m)
        self.sideLength = 0

class RobotShape:
    """Defines the geometry of the robot body and side brush, in robot coordinates."""

    def __init__(self):
        # Clockwise list of vertices of a convex robot body polygon. The last vertex is connected to the first.
        # (m,m) in robot CS.
        self.bodyPolygon = 0

        # Position (in robot CS) of side brush center (m,m)
        self.sideBrushPos = (0.05,0.15)

        # Radius of the side brush (m)
        self.sideBrushRadius = 0.05

class Robot:
    """Defines how the robot moves and interacts with obstacles."""

    def __init__(self):
        # Pose of the robot (x, y, theta) in (m, m, rad)
        self._robotPose = (0,0,0)

        # The World that the robot drives in
        self._world = World()

        # Geometry of the robot
        self._robotShape = RobotShape()

    def getRobotCenteredMap(self):
        """Return a (sampled down) bitmap of the world, in the robot CS."""

    def drive(self, curve):
        """Update the robot pose based on the drive command and return reward."""

        # Update robot pose according to the drive command

        # Compute reward: Increase if the sidebrush has cleaned something new,
        # and decrease if the robot body is in collision.
        # (Maybe decrease more, the more of the body is in collision?)
        reward = 5 * self._updateSideBrushCleaning()
        if (self._isInBodyCollision()):
            reward -= 100

        return reward

    def _updateSideBrushCleaning(self):
        # Return how many not cleaned obstacle pixels fall inside the side brush circle.
        # Also update their state in the World bitmap.
        return 0

    def _isInBodyCollision(self):
        """Return if the robot body is in collision with the World."""
        return False

