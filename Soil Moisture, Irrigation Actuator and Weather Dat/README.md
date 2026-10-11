# Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation Field Trial (Arnesano, Apulia, Italy, 2025)

**Version:** 1.0
**Date of publication:** 07/08/2026
**DOI:** [assigned by Mendeley Data on publication]
**License:** CC BY 4.0
**Contributors:** [TO BE COMPLETED — full names, ORCID, affiliation, corresponding author e-mail]
**Related publication:** a manuscript based on this dataset is under review. Its reference and
DOI will be added to this record once published.

---

## 1. Summary

This dataset documents a full growing season (February–September 2025) of an IoT-instrumented
open-field and greenhouse farm in Arnesano (Lecce, Apulia, Italy), where five cultivated sectors
were monitored with soil-moisture, pH and electrical-conductivity probes and served by
individually controlled irrigation and fertigation valves.

It contains three layers, each in its own archive:

1. **Raw layer** — untouched exports of the IoT platform: sensor measurements and actuator
   (valve) telemetry, at the native acquisition rate.
2. **Processed layer** — reconstructed and validated irrigation events, per-sector 10-minute
   time series merged with reanalysis weather data, and analysis-ready tables with
   engineered features and 24-hour soil-moisture targets.
3. **Code layer** — the ordered Python pipeline that turns the raw layer into the processed
   layer. It depends only on pandas, numpy and requests.

The dataset supports two uses that are rarely served by a single public resource: forecasting
soil moisture from real, imperfect farm telemetry, and studying how the soil responds to a
measured amount of applied water under known weather.

## 2. Value of the data

- Real actuator telemetry is published alongside sensor data, so applied water volume is a
  **measured, actionable variable** rather than an assumption. Most public soil-moisture datasets
  contain only sensor readings.
- Five sectors with **four different crop/substrate configurations** (open-field tomato, potted
  tomato, zucchini, blueberry) share the same weather forcing, enabling cross-sector transfer and
  generalisation studies.
- The raw layer is published **unmodified**, including sensor dropouts, flatlines, spikes and
  ambiguous valve signals, so that alternative cleaning strategies can be benchmarked against the
  one implemented here.
- Irrigation events are reconstructed from noisy valve signals through a documented, reproducible
  multi-phase procedure, and the resulting event list is published as a reusable intermediate
  product.
- The processed layer is directly usable for modelling: features and 24-hour-ahead targets are
  already aligned, and the forward-looking columns are flagged as such.

## 3. Experimental site and setup

| Item | Value |
|---|---|
| Site | Arnesano, Province of Lecce, Apulia, Italy |
| Coordinates | 40.349306 N, 18.081556 E |
| Monitoring period | 2025-02-09 to 2025-09-27 |
| Sectors | 5 irrigation sectors (a 6th valve/probe ID exists in the raw telemetry but was not cultivated/monitored consistently and is not part of the processed layer) |
| Sampling (raw) | Event-driven / minute-level, sensor dependent |
| Sampling (processed) | 10-minute regular grid |
| Time zone | Local time (Europe/Rome), naive timestamps, no DST correction applied |

### Sector configuration

| Sector | Crop / substrate | Soil model used | PWP | MADP | FC | SP | Nominal flow rate |
|---|---|---|---|---|---|---|---|
| 1 | Tomato, open field | `TOMATO_SOIL` | 20 | 40 | 60 | 80 | 1.1 L/min |
| 2 | Tomato, open field | `TOMATO_SOIL` | 20 | 40 | 60 | 80 | 1.1 L/min |
| 3 | Tomato, pots | `TOMATO_POT` | 30 | 60 | 80 | 90 | 1.1 L/min |
| 4 | Zucchini | `ZUCCHINI_SOIL` | 40 | 80 | 90 | 95 | 1.1 L/min |
| 5 | Blueberry | `BLUEBERRY_SOIL` | 35 | 70 | 90 | 95 | 0.4 L/min |

PWP = permanent wilting point, MADP = management allowed depletion point, FC = field capacity,
SP = saturation point. Values are expressed on the same 0–100 scale as the soil-moisture probe
output (relative volumetric water content, %). They define the five agronomic classes used
throughout the analysis:

| Class | Label | Condition |
|---|---|---|
| 0 | Danger | moisture < PWP |
| 1 | Stress | PWP ≤ moisture < MADP |
| 2 | Optimal | MADP ≤ moisture < FC |
| 3 | Excess | FC ≤ moisture < SP |
| 4 | Saturation | moisture ≥ SP |

### Sensor and actuator identifiers

| Prefix | Device | Notes |
|---|---|---|
| `SUT0000n` | Soil moisture probe, sector *n* | Value in `humidity_percentage` |
| `SNPK0000n_CE` | Soil electrical conductivity, sector *n* | µS/cm |
| `SNPK0000n_PH` | Soil pH, sector *n* | pH units |
| `STE00001` / `STI00001` | Air temperature, outside / inside greenhouse | °C |
| `SUE00001` / `SUI00001` | Air relative humidity, outside / inside greenhouse | % |
| `SV00001` | Auxiliary probe, present in raw export only | Not used in the pipeline |
| `AIRR0000n` | Irrigation valve, sector *n* | `open` = 1 open, 0 closed |
| `AFIRR00001` | Fertigation unit (shared) | Channels 1–4 = N, P, K, pH mixes |

Weather variables are **not** measured on site: they are retrieved from the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
(ERA5 reanalysis) for the site coordinates and linearly interpolated to minute resolution
before aggregation. See `03_code/01_merge_datasets.py`.

## 4. Package contents

```
README.md                      this file
DATA_DICTIONARY.md             column-by-column description of every published table
LICENSE.txt                    CC BY 4.0
CITATION.cff                   machine-readable citation metadata
file_manifest.csv              per-file size, row count and SHA-256 checksum
01_raw_data.zip                 36 MB download, 427 MB unzipped — untouched platform exports
02_processed_data.zip          6.5 MB download,  31 MB unzipped — derived, analysis-ready tables
03_code.zip                     19 KB download,  50 KB unzipped — preprocessing pipeline
```

### 01_raw_data.zip

```
raw_sensor/
  MEASUREMENT_all.csv           complete sensor export (all nodes)
  MEASUREMENT_ce.csv            electrical conductivity     } strict subsets of
  MEASUREMENT_ph.csv            pH                          } MEASUREMENT_all.csv,
  MEASUREMENT_ste.csv           outside air temperature     } split by sensor type;
  MEASUREMENT_sti.csv           inside air temperature      } these are the files
  MEASUREMENT_sue.csv           outside air humidity        } the pipeline reads
  MEASUREMENT_sui.csv           inside air humidity         }
  MEASUREMENT_sut.csv           soil moisture               }
raw_irrigation_actuators/
  raw_actuators.csv             valve open/close telemetry (3,055,944 rows)
  raw_measurements.csv          minute-level sensor telemetry stream (6,034,225 rows)
  raw_nominal_flow_rates.csv    nominal flow rate per valve and fertigation channel
```

The per-type `MEASUREMENT_*.csv` files were verified to be exact subsets of
`MEASUREMENT_all.csv` (matched on `pk_measurement`). Both are published because the pipeline
scripts read the split files, while the complete export additionally contains the `SV00001`
node. `raw_measurements.csv` is a second, minute-resampled view of the same sensor stream
produced by the platform; it is published for provenance and is not consumed by the pipeline.

### 02_processed_data.zip

```
irrigation_events/
  final_irrigation_events.csv     789 validated irrigation/fertigation events
  sector_statistics_report.csv    per-sector event statistics
merged/
  dataset_zone_{1..5}.csv         10-min regular grid, sensors + weather + water applied
                                  33,259 rows per sector, 2025-02-09 to 2025-09-27
preprocessed/
  dataset_zone_{1..5}_preprocessed.csv
                                  cleaned, feature-engineered, 24 h targets, ML-ready
```

Row counts and temporal coverage of the preprocessed files:

| File | Rows | From | To |
|---|---|---|---|
| `dataset_zone_1_preprocessed.csv` | 9,531 | 2025-02-28 15:00 | 2025-09-04 18:50 |
| `dataset_zone_2_preprocessed.csv` | 11,929 | 2025-02-28 14:20 | 2025-09-23 10:10 |
| `dataset_zone_3_preprocessed.csv` | 1,118 | 2025-02-14 13:10 | 2025-06-25 08:10 |
| `dataset_zone_4_preprocessed.csv` | 12,676 | 2025-03-01 07:10 | 2025-09-23 14:40 |
| `dataset_zone_5_preprocessed.csv` | 11,814 | 2025-02-11 09:20 | 2025-09-22 20:00 |

Sector 3 is markedly shorter than the others because its probe stopped returning valid readings
in late June; rows are dropped rather than imputed whenever soil moisture is missing beyond the
interpolation limit. Users should treat sector 3 with care in cross-sector comparisons.

### 03_code.zip

The pipeline that produces the processed layer from the raw layer. Stages run in numerical order
and each reads what the previous one wrote.

| File | Purpose |
|---|---|
| `config.py` | Every path, threshold and physical parameter, defined once |
| `00_irrigation_actuators_cleaning.py` | Reconstructs validated irrigation events from raw valve telemetry |
| `01_merge_datasets.py` | Merges sensors + events + Open-Meteo weather into per-sector 10-min tables |
| `02_preprocessing.py` | Cleaning, interpolation, calendar features, 24 h targets and water volumes |
| `requirements.txt` | pandas, numpy, requests |
| `README.md` | Running instructions and design notes |

## 5. Methods

### 5.1 Irrigation event reconstruction (`00_irrigation_actuators_cleaning.py`)

The platform records both physically observed valve transitions (`is_raw = True`) and
synthetic/inferred ones (`is_raw = False`). The latter cannot be trusted as-is, so events are
built and filtered in four phases:

1. **Event generation.** Valve open/close cycles are paired. Soil moisture at the instant of each
   transition is attached with a backward `merge_asof`. A synthetic activation is accepted only if
   it is agronomically plausible — soil moisture below the sector start threshold, or an active
   fertigation cycle. Events are constrained to the allowed irrigation window (06:00–18:00) and
   fertigation cycles are subject to a timeout.
2. **Debouncing.** Consecutive events separated by less than 2 minutes are merged, absorbing
   signal flicker.
3. **Noise filtering.** Events shorter than 15 minutes are discarded.
4. **Anomaly filtering.** Events longer than 5 hours are discarded *when* they ended by
   `time_restriction` or `physical_valve_closure`, since in those cases the length points to a
   missing closure signal rather than a real irrigation. Long events that ended by
   `fertigation_timeout` or `humidity_threshold` are retained as genuine, so 11 events in the
   published list exceed 5 hours (maximum 9.53 h).

Per-sector thresholds:

| Sector | Start threshold (irrigate below) | Stop threshold (stop above) |
|---|---|---|
| 1 | 40 | 60 |
| 2 | 40 | 60 |
| 3 | 60 | 80 |
| 4 | 80 | 90 |
| 5 | 50 | 60 |

The result is `final_irrigation_events.csv`: 789 events (432 standard irrigations, 357
fertigations) spanning 2025-02-17 to 2025-08-31.

### 5.2 Merging (`01_merge_datasets.py`)

Sensor readings are rounded to the minute, averaged per node and timestamp, mapped to their
sector, and resampled to a 10-minute grid by arithmetic mean. Validated events are expanded to a
minute-level binary valve signal, then summed within each 10-minute bin to give
`irrigation_duration_minutes` (0–10). Applied water is the product of that duration and the
sector nominal flow rate. Weather variables from Open-Meteo are interpolated to minute resolution
and joined on the same grid.

### 5.3 Preprocessing and feature engineering (`02_preprocessing.py`)

- **Physical range filtering:** applied to soil moisture only — readings outside 0–115 are
  removed and values in 100–115 are clipped to 100. The plausibility bands declared for pH (3–9)
  and EC (0–1000 µS/cm) are used as reference lines in the diagnostic plots and are **not**
  enforced as a filter, so a small fraction of implausible pH and EC readings survives into the
  preprocessed files (see §7).
- **Imputation:** time-based interpolation of soil moisture, pH and EC with a 2-step limit; rows
  still missing after that are dropped.
- **Despiking:** a 9-sample rolling filter removes single-point jumps above 15 percentage points.
- **Flatline trimming:** trailing constant soil-moisture runs, which indicate a stalled probe, are
  removed.
- **Calendar features:** month, day, hour, day period (0 = 05–12, 1 = 12–17, 2 = 17–21, 3 = 21–05)
  and quarter.
- **Targets, 24 h horizon (144 steps of 10 min):**
  `target_point_24h` = soil moisture 144 steps ahead;
  `target_mean_24h` = mean soil moisture over the following 144 steps.
- **Water features:** `water_vol_to_24h` = litres applied over the following 144 steps;
  `water_vol_past_4h` = litres applied over the preceding 24 steps.
- **Diagnostics:** the Spearman correlation between `water_vol_to_24h` and the realised moisture
  change `real_moisture_delta` is computed per sector as a physical-consistency check.

> **Steps, not hours.** The windows above are counted in rows of the cleaned series. Cleaning
> removes rows, so 144 steps span 24 hours only where the series is continuous; §7 gives the
> realised horizon per sector.

> **Leakage warning.** `target_point_24h`, `target_mean_24h` and `real_moisture_delta` are
> forward-looking by construction. `water_vol_to_24h` is also forward-looking: it is the water
> applied over the horizon, so it is a legitimate input only where the irrigation schedule is
> known in advance. Do not use it as a feature in a purely predictive setting where future
> irrigation is unknown. `real_moisture_delta` must never be used as a feature.

## 6. How to reproduce

```bash
unzip 01_raw_data.zip && unzip 03_code.zip

mkdir -p data/original
mv 01_raw_data/raw_sensor 01_raw_data/raw_irrigation_actuators data/original/
mv 03_code src

python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

python src/00_irrigation_actuators_cleaning.py   # ~2 min  -> data/irrigation_actuators/
python src/01_merge_datasets.py                  # ~3 min  -> data/merged/   (needs internet)
python src/02_preprocessing.py                   # ~10 s   -> data/preprocessed/
```

Paths are resolved from the location of `config.py`, so the scripts can be run from any working
directory. To start from the merged tables instead, unzip `02_processed_data.zip` into `data/`
and run stage 02 alone.

**Weather data.** Stage 01 queries the Open-Meteo archive at run time, and Copernicus revises
ERA5 as new observations are assimilated, so the weather columns can shift slightly between runs:
`weather_humidity`, `weather_rain`, `weather_wind_speed` and `weather_radiation` are stable, while
`weather_temp` and the two soil-temperature columns move by up to 0.1 °C and `weather_pressure` by
up to 0.09 hPa. Start from the provided `merged/` tables to avoid the query altogether.

**Requirements**

- Python 3.11 or later; three packages only (`pandas`, `numpy`, `requests`), pinned in
  `03_code/requirements.txt`.
- Stage 01 requires an internet connection. Stages 00 and 02 are fully offline.

## 7. Ethics

The dataset contains only environmental measurements and machine telemetry. It includes no
personal data and no human or animal subjects. The internal database identifiers retained in the
raw export (`pk_measurement`, `fk_iot_node`, `fk_channel`, `pk_company`) refer to devices and to
the farm organisation, not to individuals.

## 8. How to cite

> [TO BE COMPLETED — authors] (2026). *Soil Moisture, Irrigation Actuator and Weather Dataset
> from a Multi-Sector Precision-Irrigation Field Trial (Arnesano, Italy, 2025)*.
> Mendeley Data, V1. doi: [assigned on publication]

The manuscript that uses this dataset is under review; once it is published, please cite it
alongside the dataset. This record will be updated with its reference.

## 9. Licence

This dataset is released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0). You may share and adapt it for any purpose, including commercially, provided you give
appropriate credit. See `LICENSE.txt`.

Weather variables are derived from Open-Meteo historical weather data (CC BY 4.0), based on
ERA5 reanalysis by ECMWF / Copernicus Climate Change Service.
