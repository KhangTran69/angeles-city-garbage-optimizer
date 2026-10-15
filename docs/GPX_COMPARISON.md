# 📍 GPX Route Tri-Comparison Feature

**Thesis AL-1: Real-World Driver Behavior Validation**

## Overview

This feature enables scientific comparison between:
1. **Baseline (Manual)** - Actual driver routes recorded via Geo Tracker (Week 1)
2. **Optimized** - Routes from the GA optimizer (Week 2+)
3. **Local Knowledge** - Experienced drivers' preferred routes

## Data Collection Protocol

### Week 1 - Baseline Collection
1. Install **Geo Tracker** (Android/iOS) on driver phones
2. Drivers record their normal routes **without navigation assistance**
3. Export GPX files from Geo Tracker
4. Label files: `baseline_driver1_date.gpx`, `baseline_driver2_date.gpx`, etc.

### Week 2+ - Optimized Route Validation
1. Generate optimized routes from the GA system
2. Provide routes to drivers (printout or digital)
3. Record actual driven routes via Geo Tracker
4. Compare: Did drivers follow the optimized route? How much deviation?

### Local Knowledge Routes
1. Ask experienced drivers (5+ years in area) to drive their preferred route
2. Record via Geo Tracker
3. Compare against both baseline and optimized

## How to Use

### 1. Upload GPX Files
- Navigate to **📍 GPX COMPARISON** tab
- Upload up to 3 GPX files (baseline required, others optional)
- Supported: Geo Tracker, Strava, Garmin, any standard GPX 1.0/1.1

### 2. Compare Routes
- Click **🔍 COMPARE ROUTES**
- System automatically computes:
  - Total distance (km)
  - Duration (minutes)
  - Average speed
  - Number of stops
  - Estimated fuel consumption
  - CO₂ emissions

### 3. Analyze Results
- **Efficiency Ranking**: See which route performed best
- **Savings Analysis**: Distance/fuel/time saved vs baseline
- **Deviation Score**: How much optimized/local routes differ from baseline (in meters)

### 4. Export Data
- **Save Project**: JSON format for later analysis
- **Export CSV**: Metrics table for statistical analysis
- **Export Full JSON**: Complete comparison data

## Metrics Formulas

### Fuel Estimation
```
Fuel (L) = Distance (km) × 0.15 L/km
```
*Based on average garbage truck fuel consumption*

### CO₂ Estimation
```
CO₂ (kg) = Fuel (L) × 2.68 kg CO₂/L
```
*Standard diesel emission factor*

### Deviation Score
```
Deviation = Average point-to-route distance (meters)
```
*Lower score = route more similar to baseline*

### Annual Savings Projection
```
Annual Savings (₱) = Distance Saved (km) × 0.15 L/km × ₱100/L × 250 days/year
```
*Assumes 250 collection days per year*

## File Organization

```
angeles-garbage-optimizer/
├── app.py                          # Main Streamlit app (includes GPX tab)
├── src/
│   ├── gpx_utils.py                # GPX parsing + comparison logic
│   ├── logger.py                   # Structured logging
│   └── ...                         # Other modules
├── gpx_projects/                   # Saved comparison projects (auto-created)
│   └── Route_Set_April_2026.json
├── logs/                           # Application logs (auto-created)
│   └── optimizer_20260413.log
└── docs/
    └── GPX_COMPARISON.md           # This file
```

## Thesis Integration

### Chapter 3: Methodology
This feature provides **empirical validation** of your optimization algorithm:

1. **Baseline Validation**: Prove current routes are suboptimal
2. **Optimization Effectiveness**: Quantify improvements from GA
3. **Human Factor Analysis**: Do experienced drivers naturally find near-optimal routes?

### Statistical Tests
Export CSV data and run:
- **Paired t-test**: Baseline vs Optimized distances
- **ANOVA**: Compare all three route types
- **Correlation**: Driver experience vs route efficiency

### Expected Results (Hypothesis)
- Optimized routes: **15-25% shorter** than baseline
- Local knowledge routes: **10-20% shorter** than baseline
- Optimized vs Local knowledge: **5-10% difference** (optimized should win)

## Troubleshooting

### GPX File Won't Upload
- Ensure file is valid GPX 1.0 or 1.1 format
- Check file size (< 10 MB recommended)
- Try exporting from Geo Tracker with lower sampling rate

### No Timestamps in GPX
- Some GPS apps don't record timestamps
- Duration/speed metrics will be unavailable
- Distance metrics still work correctly

### Map Overlay Not Showing
- Currently in development (marked as 🚧)
- Use exported data in Google My Maps, QGIS, or similar

## Sample Data

For testing, sample GPX files are available at:
```
tests/sample_gpx/
├── baseline_sample.gpx
├── optimized_sample.gpx
└── local_knowledge_sample.gpx
```

Run tests with:
```bash
pytest tests/test_gpx_utils.py -v
```

## Future Enhancements

- [ ] Interactive map overlay (all three routes on same map)
- [ ] Heatmap visualization of deviation areas
- [ ] Automatic detection of "shortcut" segments
- [ ] Integration with OpenStreetMap for road name labeling
- [ ] Multi-day aggregation (weekly/monthly comparisons)
- [ ] Driver behavior scoring (consistency, adherence to optimized routes)

## Citation

When using this feature in your thesis:

```bibtex
@software{angeles_gpx_comparison_2026,
  title = {GPX Route Tri-Comparison Feature},
  author = {Khang Tran},
  year = {2026},
  note = {Part of Angeles City Garbage Route Optimizer - Thesis AL-1},
  url = {https://github.com/your-repo/angeles-garbage-optimizer}
}
```

---

**Status**: ✅ Production Ready (Map overlay pending)  
**Version**: 1.0.0  
**Last Updated**: 2026-04-13
