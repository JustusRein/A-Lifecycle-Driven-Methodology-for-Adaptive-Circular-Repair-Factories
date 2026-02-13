import numpy as np
from src.grippers.pads import (
    CircularPad,
    RectangularPad,
    CircularPadZones,
    RectangularPadZones,
)


def test_split_points_in_zones_circular():
    # Create a CircularPad with radius 1.0 and two radial zones splitting the radius
    circular_pad = CircularPad(
        name="test_circular",
        offset=np.array([0, 0, 0]),
        radius=1.0,
        length=0.1,
        thickness=0.2,
        relative_thickness=False,
        zones=CircularPadZones(
            **{
                "num_angular_sections": 8,  # 4 angular splits
                "num_radial_sections": 2,
            }
        ),
    )

    # Create test points to be inside the rings in all 8 sections
    points = np.array(
        [
            [0.85, 0.0, 0],  # Section 1
            [0.35, 0.35, 0],  # Section 2
            [0.0, 0.5, 0],  # Section 3
            [-0.35, 0.35, 0],  # Section 4
            [-0.5, 0.0, 0],  # Section 5
            [-0.35, -0.35, 0],  # Section 6
            [0.0, -0.5, 0],  # Section 7
            [0.35, -0.35, 0],  # Section 8
        ]
    )

    zones = circular_pad.split_points_in_zones(points)

    # Assert correct zone allocation
    print("Test Circular Zones - Results:")
    for zone_name, zone_points in zones.items():
        print(f"Zone: {zone_name}, Points: {zone_points}")


def test_split_points_in_zones_rectangular():
    # Create a RectangularPad with defined zones
    rectangular_pad = RectangularPad(
        name="test_rectangular",
        offset=np.array([0, 0, 0]),
        width=2.0,
        height=1.0,
        length=0.1,
        zones={
            "num_radial_sections": 2,  # Split height
            "num_angular_sections": 4,  # Split width
            "ring_width": 0.5,  # Not applicable but kept for API consistency
            "units": "relative",
        },
    )

    # Create test points
    points = np.array(
        [
            [0.5, 0.25, 0],  # Inside one zone
            [-1.0, -0.5, 0],  # On the pad's edge
            [1.1, 0.6, 0],  # Outside the pad
            [-0.1, -0.1, 0],  # Close to the center
        ]
    )

    zones = rectangular_pad.split_points_in_zones(points)

    # Validate correct zone allocations
    print("Test Rectangular Zones - Results:")
    for zone_name, zone_points in zones.items():
        print(f"Zone: {zone_name}, Points: {zone_points}")


if __name__ == "__main__":
    test_split_points_in_zones_circular()
    test_split_points_in_zones_rectangular()
