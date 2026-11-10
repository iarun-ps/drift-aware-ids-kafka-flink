"""Loading the CIC-IDS2017 GeneratedLabelledFlows CSV files."""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cic2017" / "GeneratedLabelledFlows" / "TrafficLabelling"

# (short name, file name), in the order the traffic was captured
DAY_FILES = [
    ("mon", "Monday-WorkingHours.pcap_ISCX.csv"),
    ("tue", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("wed", "Wednesday-workingHours.pcap_ISCX.csv"),
    ("thu_am", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("thu_pm", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("fri_am", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("fri_pm_portscan", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("fri_pm_ddos", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
]

# column names with typos in the original files -> standard names
RENAME = {
    "FwdPacket Length Max": "Fwd Packet Length Max",
    "BwdPacket Length Mean": "Bwd Packet Length Mean",
    "BwdHeader Length": "Bwd Header Length",
}

DUP_COL = "Fwd Header Length.1"


def load_day(name: str) -> pd.DataFrame:
    """Load one capture file by its short name (e.g. 'tue') and tidy it."""
    files = dict(DAY_FILES)
    path = DATA_DIR / files[name]
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME)

    # the file contains 'Fwd Header Length' twice; drop the copy if identical
    if DUP_COL in df.columns and df[DUP_COL].equals(df["Fwd Header Length"]):
        df = df.drop(columns=DUP_COL)

    df["Label"] = (
        df["Label"].astype(str)
        .str.replace(r"[^\x20-\x7E]+", "-", regex=True)
        .str.strip()
    )

    # times are stored in 12-hour format without AM/PM; the capture ran from
    # about 08:00 to 17:00, so hours before 8 are really afternoon (add 12 h)
    ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df["Timestamp"] = ts + pd.to_timedelta((ts.dt.hour < 8) * 12, unit="h")

    df["row_order"] = np.arange(len(df))
    df["source_file"] = path.name
    return df
