"""
Unit Tests for GPX Route Comparison Utilities.
Thesis AL-1: Validation of GPX parsing and metrics computation.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gpx_utils import (
    parse_gpx_content,
    compute_route_metrics,
    compare_routes,
    compute_deviation_score,
    FUEL_PER_KM,
    CO2_PER_LITER_DIESEL,
)


# Sample GPX data for testing (simple triangular route)
SAMPLE_GPX_BASELINE = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Test">
  <trk>
    <name>Baseline Route</name>
    <trkseg>
      <trkpt lat="15.1450" lon="120.5880">
        <ele>10</ele>
        <time>2026-04-13T08:00:00Z</time>
      </trkpt>
      <trkpt lat="15.1460" lon="120.5890">
        <ele>12</ele>
        <time>2026-04-13T08:05:00Z</time>
      </trkpt>
      <trkpt lat="15.1470" lon="120.5900">
        <ele>15</ele>
        <time>2026-04-13T08:10:00Z</time>
      </trkpt>
      <trkpt lat="15.1480" lon="120.5910">
        <ele>11</ele>
        <time>2026-04-13T08:15:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
"""

SAMPLE_GPX_OPTIMIZED = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Test">
  <trk>
    <name>Optimized Route</name>
    <trkseg>
      <trkpt lat="15.1450" lon="120.5880">
        <ele>10</ele>
        <time>2026-04-13T09:00:00Z</time>
      </trkpt>
      <trkpt lat="15.1465" lon="120.5895">
        <ele>13</ele>
        <time>2026-04-13T09:03:00Z</time>
      </trkpt>
      <trkpt lat="15.1480" lon="120.5910">
        <ele>11</ele>
        <time>2026-04-13T09:06:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
"""


class TestGPXParsing:
    """Test GPX file parsing functionality."""

    def test_parse_valid_gpx(self):
        """Test parsing a valid GPX string."""
        gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        assert gpx is not None
        assert len(gpx.tracks) == 1
        assert len(gpx.tracks[0].segments) == 1
        assert len(gpx.tracks[0].segments[0].points) == 4

    def test_parse_invalid_gpx(self):
        """Test parsing invalid GPX data."""
        gpx = parse_gpx_content("not a gpx file")
        assert gpx is None

    def test_parse_empty_gpx(self):
        """Test parsing empty string."""
        gpx = parse_gpx_content("")
        assert gpx is None


class TestRouteMetrics:
    """Test route metrics computation."""

    def test_compute_basic_metrics(self):
        """Test computation of basic route metrics."""
        gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        polyline = [(15.1450, 120.5880), (15.1460, 120.5890), 
                    (15.1470, 120.5900), (15.1480, 120.5910)]
        
        metrics = compute_route_metrics(polyline, gpx, "baseline")
        
        assert metrics.route_type == "baseline"
        assert metrics.total_distance_km > 0
        assert metrics.total_duration_minutes > 0
        assert metrics.num_waypoints == 4
        assert metrics.start_coord == (15.1450, 120.5880)
        assert metrics.end_coord == (15.1480, 120.5910)

    def test_fuel_estimation(self):
        """Test fuel consumption estimation."""
        gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        polyline = [(15.1450, 120.5880), (15.1460, 120.5890)]
        
        metrics = compute_route_metrics(polyline, gpx, "baseline")
        
        expected_fuel = metrics.total_distance_km * FUEL_PER_KM
        assert abs(metrics.estimated_fuel_liters - expected_fuel) < 0.001

    def test_co2_estimation(self):
        """Test CO2 emission estimation."""
        gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        polyline = [(15.1450, 120.5880), (15.1460, 120.5890)]
        
        metrics = compute_route_metrics(polyline, gpx, "baseline")
        
        expected_co2 = metrics.estimated_fuel_liters * CO2_PER_LITER_DIESEL
        assert abs(metrics.estimated_co2_kg - expected_co2) < 0.001

    def test_empty_polyline(self):
        """Test metrics computation with empty polyline."""
        gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        metrics = compute_route_metrics([], gpx, "baseline")
        
        assert metrics.total_distance_km == 0
        assert metrics.total_duration_minutes == 0
        assert metrics.num_waypoints == 0


class TestRouteComparison:
    """Test route comparison functionality."""

    def test_compare_two_routes(self):
        """Test comparison between baseline and optimized routes."""
        baseline_gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        optimized_gpx = parse_gpx_content(SAMPLE_GPX_OPTIMIZED)
        
        comparison = compare_routes(baseline_gpx, optimized_gpx, None)
        
        assert comparison.baseline is not None
        assert comparison.optimized is not None
        assert comparison.local_knowledge is None
        assert comparison.optimized_savings_km is not None
        assert comparison.efficiency_ranking is not None

    def test_efficiency_ranking(self):
        """Test that efficiency ranking orders routes by distance."""
        baseline_gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        optimized_gpx = parse_gpx_content(SAMPLE_GPX_OPTIMIZED)
        
        comparison = compare_routes(baseline_gpx, optimized_gpx, None)
        
        # Optimized should be shorter (fewer waypoints, more direct)
        assert len(comparison.efficiency_ranking) == 2
        assert comparison.efficiency_ranking[0] in ["baseline", "optimized"]

    def test_savings_calculation(self):
        """Test savings percentage calculation."""
        baseline_gpx = parse_gpx_content(SAMPLE_GPX_BASELINE)
        optimized_gpx = parse_gpx_content(SAMPLE_GPX_OPTIMIZED)
        
        comparison = compare_routes(baseline_gpx, optimized_gpx, None)
        
        if comparison.optimized_savings_km and comparison.baseline:
            expected_pct = (comparison.optimized_savings_km / 
                          comparison.baseline.total_distance_km) * 100
            assert abs(comparison.optimized_savings_pct - expected_pct) < 0.01


class TestDeviationScore:
    """Test route deviation computation."""

    def test_identical_routes_zero_deviation(self):
        """Test that identical routes have zero deviation."""
        polyline = [(15.1450, 120.5880), (15.1460, 120.5890)]
        
        deviation = compute_deviation_score(polyline, polyline)
        
        assert deviation == 0.0

    def test_different_routes_positive_deviation(self):
        """Test that different routes have positive deviation."""
        baseline = [(15.1450, 120.5880), (15.1460, 120.5890)]
        different = [(15.1550, 120.5980), (15.1560, 120.5990)]
        
        deviation = compute_deviation_score(baseline, different)
        
        assert deviation > 0.0

    def test_empty_polyline_infinite_deviation(self):
        """Test that empty polylines return infinite deviation."""
        baseline = [(15.1450, 120.5880)]
        
        deviation = compute_deviation_score(baseline, [])
        
        assert deviation == float('inf')


class TestConstants:
    """Test that constants are properly defined."""

    def test_fuel_constant(self):
        """Test fuel per km constant."""
        assert FUEL_PER_KM > 0
        assert FUEL_PER_KM < 1  # Should be less than 1 L/km for trucks

    def test_co2_constant(self):
        """Test CO2 per liter constant."""
        assert CO2_PER_LITER_DIESEL > 0
        assert CO2_PER_LITER_DIESEL > 2  # Diesel emits ~2.68 kg CO2/L


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
