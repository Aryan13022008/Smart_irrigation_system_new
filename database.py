"""
Smart Agriculture - Persistent Sensor History & Calibration Database
=====================================================================
Manages local SQLite database for timestamped ESP32 sensor telemetry and 
independent per-probe soil ADC calibration settings.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "sensor_history.db"


def get_db_path(custom_path=None):
    if custom_path is not None:
        return Path(custom_path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_PATH


def get_connection(db_path=None):
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for thread safety and high read/write concurrency
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.Error:
        pass
    return conn


def init_db(db_path=None):
    """
    Initializes SQLite tables for persistent sensor readings and probe calibrations.
    """
    path = get_db_path(db_path)
    conn = get_connection(path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT UNIQUE NOT NULL,
                    recorded_at REAL NOT NULL,
                    soil1_raw REAL,
                    soil2_raw REAL,
                    water_level_raw REAL,
                    rain_pin_raw REAL,
                    rain_detected INTEGER,
                    temperature_c REAL,
                    humidity_pct REAL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_readings_recorded_at 
                ON sensor_readings(recorded_at);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS probe_calibration (
                    probe_id TEXT PRIMARY KEY,
                    dry_adc REAL NOT NULL,
                    wet_adc REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
    finally:
        conn.close()


def log_reading(data, source="esp32_usb", status="valid", db_path=None, timestamp=None, recorded_at=None):
    """
    Records genuine sensor readings in SQLite.
    
    Guards:
    - Rejects non-dict or empty data.
    - Rejects payloads with no genuine numeric readings.
    - Prevents crashes on duplicate timestamps (INSERT OR IGNORE).
    - Catches and logs all database exceptions safely.
    """
    if not isinstance(data, dict):
        return False, "Data must be a dictionary."

    def extract_float(key):
        val = data.get(key)
        if val is None or isinstance(val, bool):
            return None
        try:
            f = float(val)
            return f
        except (ValueError, TypeError):
            return None

    soil1 = extract_float("soil1_raw")
    soil2 = extract_float("soil2_raw")
    water_level = extract_float("water_level_raw")
    rain_pin = extract_float("rain_pin_raw")
    temp_c = extract_float("temperature_c")
    hum_pct = extract_float("humidity_pct")

    # Rain detected boolean / int
    rain_val = data.get("rain_detected")
    rain_int = None
    if isinstance(rain_val, bool):
        rain_int = 1 if rain_val else 0
    elif isinstance(rain_val, (int, float)):
        rain_int = 1 if rain_val else 0

    # Ensure at least one genuine measurement exists
    measurements = [soil1, soil2, water_level, rain_pin, temp_c, hum_pct, rain_int]
    if all(m is None for m in measurements):
        return False, "Discarded: no genuine measurements in payload."

    now_epoch = time.time() if recorded_at is None else float(recorded_at)
    if timestamp is None:
        iso_ts = datetime.now(timezone.utc).isoformat()
    else:
        iso_ts = str(timestamp)

    path = get_db_path(db_path)
    init_db(path)

    try:
        conn = get_connection(path)
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO sensor_readings (
                    timestamp, recorded_at, soil1_raw, soil2_raw,
                    water_level_raw, rain_pin_raw, rain_detected,
                    temperature_c, humidity_pct, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                iso_ts, now_epoch, soil1, soil2,
                water_level, rain_pin, rain_int,
                temp_c, hum_pct, str(source), str(status)
            ))
        conn.close()
        return True, None
    except sqlite3.Error as exc:
        return False, f"Database error: {exc}"
    except Exception as exc:
        return False, f"Unexpected error logging reading: {exc}"


def get_history(limit_hours=24, db_path=None):
    """
    Safely queries genuine sensor readings from the last `limit_hours` hours.
    Returns a pandas DataFrame sorted by timestamp ascending.
    """
    path = get_db_path(db_path)
    if not path.exists():
        return pd.DataFrame()

    init_db(path)
    try:
        conn = get_connection(path)
        if limit_hours is not None:
            cutoff_epoch = time.time() - (float(limit_hours) * 3600.0)
            query = """
                SELECT timestamp, recorded_at, soil1_raw, soil2_raw,
                       water_level_raw, rain_detected, temperature_c,
                       humidity_pct, source, status
                FROM sensor_readings
                WHERE recorded_at >= ?
                ORDER BY recorded_at ASC;
            """
            df = pd.read_sql_query(query, conn, params=(cutoff_epoch,))
        else:
            query = """
                SELECT timestamp, recorded_at, soil1_raw, soil2_raw,
                       water_level_raw, rain_detected, temperature_c,
                       humidity_pct, source, status
                FROM sensor_readings
                ORDER BY recorded_at ASC;
            """
            df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_history_span(db_path=None, continuous=False, max_gap_seconds=120.0):
    """
    Returns (earliest_epoch, latest_epoch, count, span_seconds).
    
    If continuous=True, calculates the span of the most recent continuous
    logging session without gaps exceeding max_gap_seconds (default: 120s).
    """
    path = get_db_path(db_path)
    if not path.exists():
        return None, None, 0, 0.0

    init_db(path)
    try:
        conn = get_connection(path)
        cur = conn.cursor()
        if not continuous:
            cur.execute("""
                SELECT MIN(recorded_at), MAX(recorded_at), COUNT(*)
                FROM sensor_readings
                WHERE status = 'valid';
            """)
            row = cur.fetchone()
            conn.close()
            if not row or row[0] is None or row[2] == 0:
                return None, None, 0, 0.0
            earliest, latest, count = row[0], row[1], row[2]
            span = float(latest - earliest)
            return earliest, latest, count, span
        else:
            cur.execute("""
                SELECT recorded_at
                FROM sensor_readings
                WHERE status = 'valid'
                ORDER BY recorded_at DESC;
            """)
            rows = [r[0] for r in cur.fetchall()]
            conn.close()
            if not rows:
                return None, None, 0, 0.0
            latest = rows[0]
            earliest = rows[0]
            count = 1
            for i in range(len(rows) - 1):
                if rows[i] - rows[i + 1] > float(max_gap_seconds):
                    earliest = rows[i]
                    break
                earliest = rows[i + 1]
                count += 1
            span = float(latest - earliest)
            return earliest, latest, count, span
    except Exception:
        return None, None, 0, 0.0


def get_lag_reading(target_seconds_ago=3600, tolerance_seconds=900, current_time=None, db_path=None, require_continuous=False, max_gap_seconds=120.0):
    """
    Safely retrieves a genuine reading approximately `target_seconds_ago` before `current_time`.
    
    Default: 1-hour lag (3600s) with 15-minute tolerance (900s).
    A genuine reading must fall within [3600 - 900, 3600 + 900] = [2700s, 4500s] ago.
    
    If `require_continuous=True`, verifies that the reading belongs to the same continuous
    logging session (gaps <= max_gap_seconds) as current_time.
    
    Returns:
    - dict of the reading closest to the target lag.
    - None if no genuine historical reading exists within the window.
    
    Crucial: Never fabricates or interpolates synthetic values.
    """
    path = get_db_path(db_path)
    if not path.exists():
        return None

    init_db(path)
    ref_time = time.time() if current_time is None else float(current_time)
    min_epoch = ref_time - float(target_seconds_ago) - float(tolerance_seconds)
    max_epoch = ref_time - float(target_seconds_ago) + float(tolerance_seconds)
    target_epoch = ref_time - float(target_seconds_ago)

    try:
        conn = get_connection(path)
        cur = conn.cursor()

        if require_continuous:
            cur.execute("""
                SELECT recorded_at
                FROM sensor_readings
                WHERE status = 'valid' AND recorded_at <= ?
                ORDER BY recorded_at DESC;
            """, (ref_time,))
            rows_c = [r[0] for r in cur.fetchall()]
            if not rows_c:
                conn.close()
                return None
            session_start = rows_c[0]
            for i in range(len(rows_c) - 1):
                if rows_c[i] - rows_c[i + 1] > float(max_gap_seconds):
                    session_start = rows_c[i]
                    break
                session_start = rows_c[i + 1]
            min_required_span = float(target_seconds_ago) - float(tolerance_seconds)
            if (ref_time - session_start) < min_required_span:
                conn.close()
                return None
            if min_epoch < session_start:
                min_epoch = session_start

        cur.execute("""
            SELECT timestamp, recorded_at, soil1_raw, soil2_raw,
                   water_level_raw, rain_detected, temperature_c,
                   humidity_pct, source, status
            FROM sensor_readings
            WHERE recorded_at BETWEEN ? AND ?
              AND status = 'valid'
            ORDER BY ABS(recorded_at - ?) ASC
            LIMIT 1;
        """, (min_epoch, max_epoch, target_epoch))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Soil Probe Calibration Management
# ---------------------------------------------------------------------------

def validate_calibration_references(dry_adc, wet_adc):
    """
    Validates dry and wet ADC reference values for a capacitive soil probe.
    
    Rules:
    1. Both must be numeric.
    2. Both must be within valid 12-bit ADC range [0, 4095].
    3. Dry and Wet references cannot be identical (prevents zero division).
    4. For capacitive soil probes, Dry ADC (in air, high resistance/frequency)
       must be strictly greater than Wet ADC (submerged in water, low resistance).
    """
    try:
        dry = float(dry_adc)
        wet = float(wet_adc)
    except (ValueError, TypeError):
        return False, None, None, "Reference values must be numeric numbers."

    if not (0.0 <= dry <= 4095.0):
        return False, None, None, f"Dry ADC ({dry:.1f}) is out of 12-bit range [0, 4095]."

    if not (0.0 <= wet <= 4095.0):
        return False, None, None, f"Wet ADC ({wet:.1f}) is out of 12-bit range [0, 4095]."

    if dry == wet:
        return False, None, None, "Dry and Wet reference ADC values cannot be identical (would cause division by zero)."

    if dry <= wet:
        return False, None, None, (
            f"Invalid reference relationship: Dry ADC ({dry:.0f}) must be strictly greater than Wet ADC ({wet:.0f}). "
            "Capacitive probes output higher ADC in air (dry) and lower ADC submerged in water (wet)."
        )

    # Minimum spread check to prevent hyper-sensitive calibration
    if (dry - wet) < 200.0:
        return False, None, None, (
            f"Reference spread is too narrow ({dry - wet:.0f} ADC counts). "
            "Typical dry-to-wet spread for capacitive soil probes is at least 500–2000 ADC counts."
        )

    return True, dry, wet, None


def save_probe_calibration(probe_id, dry_adc, wet_adc, db_path=None):
    """
    Saves validated calibration reference values for a specific probe.
    Does NOT affect the other probe.
    """
    if probe_id not in ("probe1", "probe2"):
        return False, f"Invalid probe identifier: '{probe_id}'. Must be 'probe1' or 'probe2'."

    is_valid, dry, wet, error = validate_calibration_references(dry_adc, wet_adc)
    if not is_valid:
        return False, error

    path = get_db_path(db_path)
    init_db(path)

    try:
        conn = get_connection(path)
        with conn:
            conn.execute("""
                INSERT INTO probe_calibration (probe_id, dry_adc, wet_adc, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(probe_id) DO UPDATE SET
                    dry_adc = excluded.dry_adc,
                    wet_adc = excluded.wet_adc,
                    updated_at = excluded.updated_at;
            """, (
                probe_id, dry, wet, datetime.now(timezone.utc).isoformat()
            ))
        conn.close()
        return True, None
    except sqlite3.Error as exc:
        return False, f"Database error saving calibration: {exc}"
    except Exception as exc:
        return False, f"Unexpected error: {exc}"


def get_probe_calibration(probe_id, db_path=None):
    """
    Retrieves calibration reference for a specific probe.
    Returns:
    {
        "probe_id": str,
        "calibrated": bool,
        "dry_adc": float or None,
        "wet_adc": float or None,
        "updated_at": str or None
    }
    
    Crucial: Does NOT assign default calibration constants.
    """
    path = get_db_path(db_path)
    if not path.exists():
        return {
            "probe_id": probe_id,
            "calibrated": False,
            "dry_adc": None,
            "wet_adc": None,
            "updated_at": None,
        }

    init_db(path)
    try:
        conn = get_connection(path)
        cur = conn.cursor()
        cur.execute("""
            SELECT probe_id, dry_adc, wet_adc, updated_at
            FROM probe_calibration
            WHERE probe_id = ?;
        """, (probe_id,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return {
                "probe_id": probe_id,
                "calibrated": False,
                "dry_adc": None,
                "wet_adc": None,
                "updated_at": None,
            }
        return {
            "probe_id": row["probe_id"],
            "calibrated": True,
            "dry_adc": float(row["dry_adc"]),
            "wet_adc": float(row["wet_adc"]),
            "updated_at": row["updated_at"],
        }
    except Exception:
        return {
            "probe_id": probe_id,
            "calibrated": False,
            "dry_adc": None,
            "wet_adc": None,
            "updated_at": None,
        }


def get_all_calibrations(db_path=None):
    return {
        "probe1": get_probe_calibration("probe1", db_path=db_path),
        "probe2": get_probe_calibration("probe2", db_path=db_path),
    }


def calculate_calibrated_moisture(raw_adc, dry_adc, wet_adc):
    """
    Computes soil moisture percentage (0.0% to 100.0%) from raw ADC count.
    
    Formula:
        Moisture % = clamp(0.0, 100.0, (dry_adc - raw_adc) / (dry_adc - wet_adc) * 100.0)
        
    Returns:
    - float percentage in [0.0, 100.0]
    - None if uncalibrated or inputs are invalid
    """
    if raw_adc is None or dry_adc is None or wet_adc is None:
        return None
    try:
        raw = float(raw_adc)
        dry = float(dry_adc)
        wet = float(wet_adc)
    except (ValueError, TypeError):
        return None

    if dry <= wet:
        return None

    pct = ((dry - raw) / (dry - wet)) * 100.0
    return max(0.0, min(100.0, float(pct)))
