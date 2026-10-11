# Preprocessing pipeline

The three stages that turn the raw exports in `01_raw_data/` into the processed layer.

```
config.py                            all paths, thresholds and physical parameters
00_irrigation_actuators_cleaning.py  raw valve telemetry  -> validated irrigation events
01_merge_datasets.py                 sensors + events + weather -> merged per-sector tables
02_preprocessing.py                  merged tables -> analysis-ready tables
requirements.txt                     pandas, numpy, requests
```

Every constant lives in `config.py`, so no quantity is defined twice. Each stage reads what the
previous one wrote.

## Running

```bash
unzip 01_raw_data.zip && unzip 03_code.zip

mkdir -p data/original
mv 01_raw_data/raw_sensor 01_raw_data/raw_irrigation_actuators data/original/
mv 03_code src

python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

python src/00_irrigation_actuators_cleaning.py   # ~2 min   -> data/irrigation_actuators/
python src/01_merge_datasets.py                  # ~3 min   -> data/merged/
python src/02_preprocessing.py                   # ~10 s    -> data/preprocessed/
```

The scripts resolve their paths from the location of `config.py`, so they can be run from any
working directory. Stage 01 needs an internet connection; the other two are fully offline.

To start from the merged tables instead, unzip `02_processed_data.zip` into `data/` and run
stage 02 alone.

## Weather data

Stage 01 queries the Open-Meteo archive at run time, and Copernicus revises ERA5 as new
observations are assimilated, so the weather columns can shift slightly between runs.
`weather_humidity`, `weather_rain`, `weather_wind_speed` and `weather_radiation` are stable;
`weather_temp` and the two soil-temperature columns move by up to 0.1 °C and `weather_pressure`
by up to 0.09 hPa. Sensor and irrigation columns are unaffected. Start from the provided merged
tables to avoid the query altogether.

## Design notes

**Cleaning drops rows instead of imputing them.** Soil moisture is the modelled quantity, so
imputed values would propagate into both the features and the targets. Only gaps of at most two
samples (20 minutes) in the soil sensors are interpolated; anything longer is left missing.

**Windows are computed on the grid, targets are then restricted to clean rows.** Discarded
readings are blanked on the grid before the windows are taken, so a dropped sample never
contributes to a target, and a window that straddles a gap is still measured in hours.

**The mean target tolerates partial coverage; the point target does not.** The probe samples
intermittently — monthly coverage runs from 3 % in February to 93 % in August — so requiring a
fully sampled 24-hour window would leave no usable row in any sector. `MIN_WINDOW_COVERAGE`
(0.5) requires at least half the window to be sampled for `target_mean_24h`.
`target_point_24h` has no equivalent tolerance: the sample at t + 24 h either exists or it does
not.

**Irrigation is reconstructed, not measured.** The platform reports valve states, most of them
synthetic (`is_raw = False`). Stage 00 accepts a synthetic opening only when the soil was dry
enough to justify it or a fertigation cycle was running, then debounces flicker, drops events
shorter than 15 minutes, and drops events longer than 5 hours whose stop reason indicates a lost
closure signal. Applied water is duration times a nominal flow rate, not a flow-meter reading.

## Modifying the pipeline

Common changes, all in `config.py`:

| Goal | Change |
|---|---|
| Different prediction horizon | `TARGET_HORIZON_HOURS` |
| Accept sparser or denser target windows | `MIN_WINDOW_COVERAGE` |
| Stricter or looser spike filter | `SPIKE_WINDOW`, `SPIKE_THRESHOLD` |
| Keep more data through longer gaps | `INTERPOLATION_LIMIT` |
| Different sensor plausibility bands | `SENSOR_VALID_RANGES`, `MOISTURE_VALID_RANGE` |
| Different agronomic thresholds | `SOIL_TYPES`, `SECTOR_SOIL_TYPE` |
| Different event validation | `THRESHOLD_START`, `THRESHOLD_STOP`, `MIN_DURATION_MINUTES`, `MAX_DURATION_HOURS` |

Changing `TARGET_HORIZON_HOURS` renames the target columns accordingly, since the horizon is part
of each column name.
