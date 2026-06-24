#============================================READING==========================================================
import pandas as pd

sheet_id = "1gncQ2QAq4hy_m5ke2Fl0-WoH2JPfCqBTGIGsvClT_wo"
gid = "860237478"

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

df = pd.read_csv(url)
#======================VERIFY======================================


print("SUCCESS")
print("Rows =", len(df))
print("Columns =", list(df.columns))

print("SUCCESS")
print("Rows =", len(df))
print("Columns =", list(df.columns))

#================================================THE MAIN BODY==================================================
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 19:16:02 2026

@author: gitud
"""
# -*- coding: utf-8 -*-
"""
Smart Plug – Habit Learning with DBSCAN
Multiple clusters with independent scheduling
Confidence threshold = 0.8
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
TIME_TOLERANCE_MIN = 10          # \u00b1 minutes
MIN_SAMPLES = 5
CONFIDENCE_THRESHOLD = 0.79
MINUTES_IN_DAY = 1440

# -----------------------------
# Load dataset
# -----------------------------


# -----------------------------
# Circular time encoding
# -----------------------------
def circular_features(minutes):
    theta = 2 * np.pi * minutes / MINUTES_IN_DAY
    return np.cos(theta), np.sin(theta)

df["cos_time"], df["sin_time"] = zip(
    *df["time_of_day_min"].apply(circular_features)
)

# -----------------------------
# DBSCAN eps from \u00b1X minutes
# -----------------------------
eps = 2 * np.sin(np.pi * TIME_TOLERANCE_MIN / MINUTES_IN_DAY)

# -----------------------------
# Filter data (example: Weekday ON)


# -----------------------------
points = df[
    (df["day_type"] == "Weekday") &
    (df["event_type"] == "ON")
][["cos_time", "sin_time"]].values

# -----------------------------
# Run DBSCAN
# -----------------------------
db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
labels = db.fit_predict(points)
# ---------------------------------
# Attach ON cluster labels to df
# ---------------------------------

on_mask = (
    (df["day_type"] == "Weekday") &
    (df["event_type"] == "ON")
)

df.loc[on_mask, "on_cluster"] = labels
df["on_cluster"] = df["on_cluster"].fillna(-1).astype(int)


def plot_circular_clusters(points, labels):
    """
    Visualize DBSCAN clusters on a circular (24-hour) plot
    with time-of-day labels instead of angles
    """

    # Convert (cos, sin) \u2192 angle
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles = np.mod(angles, 2 * np.pi)

    unique_labels = np.unique(labels)

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for label in unique_labels:
        mask = labels == label
        cluster_angles = angles[mask]

        if label == -1:
            ax.scatter(
                cluster_angles,
                np.ones_like(cluster_angles),
                alpha=0.4,
                label="Noise"
            )
        else:
            ax.scatter(
                cluster_angles,
                np.ones_like(cluster_angles),
                label=f"Cluster {label}"
            )

            # Mean direction (learned schedule)
            mean_x = np.mean(points[mask, 0])
            mean_y = np.mean(points[mask, 1])
            mean_angle = np.arctan2(mean_y, mean_x)
            mean_angle = np.mod(mean_angle, 2 * np.pi)

            ax.plot(
                [mean_angle, mean_angle],
                [0, 1.2],
                linestyle="--"
            )

    # -----------------------------
    # Time-of-day axis formatting
    # -----------------------------
    hour_ticks = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    hour_labels = [f"{h:02d}:00" for h in range(24)]

    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels)

    ax.set_theta_zero_location("N")   # 00:00 at top
    ax.set_theta_direction(-1)        # Clockwise
    ax.set_rticks([])

    ax.set_title("ON Event Clusters (Time of Day)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.show()


plot_circular_clusters(points, labels)

# -----------------------------
# Extract schedules from clusters
# -----------------------------
def extract_schedules(points, labels):
    schedules = []

    # Compute cluster sizes (ignore noise)
    cluster_sizes = {
        l: np.sum(labels == l)
        for l in np.unique(labels)
        if l != -1
    }

    if not cluster_sizes:
        return schedules

    max_cluster_size = max(cluster_sizes.values())
    total_points = len(points)

    print("\nCluster confidence report:")
    print("-" * 50)

    for label, cluster_size in cluster_sizes.items():
        cluster = points[labels == label]

        # Frequency relative to dominant habit
        frequency = cluster_size / max_cluster_size

        # Mean resultant vector
        mean_x = np.mean(cluster[:, 0])
        mean_y = np.mean(cluster[:, 1])
        R = np.sqrt(mean_x**2 + mean_y**2)
        R = np.clip(R, 1e-6, 1.0)

        # Circular statistics
        sigma = np.sqrt(-2 * np.log(R))
        consistency = np.exp(-sigma)

        confidence = frequency * consistency

        # Convert mean direction \u2192 time
        mean_angle = np.arctan2(mean_y, mean_x)
        mean_minutes = (mean_angle * 1440 / (2 * np.pi)) % 1440
        hh = int(mean_minutes // 60)
        mm = int(mean_minutes % 60)

        status = "ACCEPTED" if confidence >= CONFIDENCE_THRESHOLD else "REJECTED"

        print(
            f"Cluster {label:>2} | "
            f"samples={cluster_size:>2} | "
            f"time\u2248{hh:02d}:{mm:02d} | "
            f"frequency={frequency:.2f} | "
            f"consistency={consistency:.2f} | "
            f"confidence={confidence:.2f} \u2192 {status}"
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            schedules.append({
                "cluster_id": int(label),
                "mean_minutes": mean_minutes,
                "confidence": confidence,
                "samples": cluster_size
            })

    print("-" * 50)
    return schedules


# -----------------------------
# Generate schedules
# -----------------------------
schedules = extract_schedules(points, labels)
on_schedules = schedules

# -----------------------------
# Display results
# -----------------------------
if not schedules:
    print("No reliable ON schedules learned.")
else:
    print("Learned ON schedules:")
    for s in schedules:
        hh = int(s["mean_minutes"] // 60)
        mm = int(s["mean_minutes"] % 60)

        print(
            f" \u2022 Cluster {s['cluster_id']}: "
            f"ON at {hh:02d}:{mm:02d} | "
            f"confidence={s['confidence']:.2f} | "
            f"samples={s['samples']}"
        )


# ======================================================
# Parameters for OFF learning
# ======================================================

DURATION_TOLERANCE_MIN = 10
DURATION_MIN_SAMPLES = 3
DURATION_CONFIDENCE_THRESHOLD = 0.8
CONSISTENCY_SCALE = 15  # minutes

# ======================================================
# 1. Pair ON \u2192 OFF events
# ======================================================

df_sorted = df.sort_values("timestamp").reset_index(drop=True)

pairs = []

for i in range(len(df_sorted) - 1):

    if df_sorted.loc[i, "event_type"] == "ON" and \
       df_sorted.loc[i+1, "event_type"] == "OFF":

        on_time = df_sorted.loc[i, "time_of_day_min"]
        off_time = df_sorted.loc[i+1, "time_of_day_min"]

        duration = (off_time - on_time) % MINUTES_IN_DAY

        pairs.append({
            "on_cluster": df_sorted.loc[i, "on_cluster"],
            "duration": duration
        })

duration_df = pd.DataFrame(pairs)

print("\nTotal ON\u2192OFF pairs:", len(duration_df))

# ======================================================
# 2. Duration confidence (linear)
# ======================================================

def duration_confidence(cluster_vals, all_vals):

    cluster_size = len(cluster_vals)
    total = len(all_vals)

    # Relative dominance
    frequency = cluster_size / total

    sigma = np.std(cluster_vals)
    consistency = np.exp(-sigma / CONSISTENCY_SCALE)

    return frequency * consistency

# ======================================================
# 3. Learn OFF schedules per ON cluster
# ======================================================

off_schedules = {}

#for on_cluster in sorted(duration_df["on_cluster"].unique()):
accepted_clusters = [s["cluster_id"] for s in on_schedules]
for on_cluster in accepted_clusters:

    if on_cluster == -1:
        continue

    durations = duration_df[
        duration_df["on_cluster"] == on_cluster
    ]["duration"].values

    if len(durations) < DURATION_MIN_SAMPLES:
        continue

    X = durations.reshape(-1, 1)

    db = DBSCAN(
        eps=DURATION_TOLERANCE_MIN,
        min_samples=DURATION_MIN_SAMPLES
    )

    labels_dur = db.fit_predict(X)

    print(f"\nON Cluster {on_cluster} duration analysis:")

    schedules = []

    for label in np.unique(labels_dur):

        if label == -1:
            continue

        cluster_vals = durations[labels_dur == label]

        mean_duration = np.mean(cluster_vals)

        conf = duration_confidence(
            pd.Series(cluster_vals),
            pd.Series(durations)
        )

        print(
            f"  Duration cluster {label}: "
            f"mean={mean_duration:.1f} min | "
            f"samples={len(cluster_vals)} | "
            f"confidence={conf:.2f}"
        )

        if conf >= DURATION_CONFIDENCE_THRESHOLD:
            schedules.append({
                "mean_duration": mean_duration,
                "confidence": conf,
                "samples": len(cluster_vals)
            })

    if schedules:
        off_schedules[on_cluster] = schedules

# ======================================================
# 4. Display OFF schedules
# ======================================================

print("\nLearned OFF schedules:")

if not off_schedules:
    print("No reliable OFF schedules learned.")
else:
    for k, v in off_schedules.items():
        for s in v:
            print(
                f"ON Cluster {k} \u2192 OFF after {int(s['mean_duration'])} min "
                f"(confidence={s['confidence']:.2f}, samples={s['samples']})"
            )

# ======================================================
# 5. Visualization (duration distributions)
# ======================================================

for on_cluster in off_schedules.keys():

    vals = duration_df[
        duration_df["on_cluster"] == on_cluster
    ]["duration"]

    plt.figure()
    plt.hist(vals, bins=10)
    plt.title(f"Duration Distribution \u2013 ON Cluster {on_cluster}")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Count")
    plt.show()
    #==============================final output==============================================================
print("ON schedules =", on_schedules)
print("OFF schedules =", off_schedules)

for on_schedule in on_schedules:

    hh = int(on_schedule["mean_minutes"] // 60)
    mm = int(on_schedule["mean_minutes"] % 60)

    cluster_id = on_schedule["cluster_id"]

    if cluster_id in off_schedules:

        for off_schedule in off_schedules[cluster_id]:

            duration = int(off_schedule["mean_duration"])

            print(f"{hh}:{mm:02d} {duration}")

#============================result to sheet========================================================================
import requests

all_data = []

for on_schedule in on_schedules:

    hh = int(on_schedule["mean_minutes"] // 60)
    mm = int(on_schedule["mean_minutes"] % 60)

    cluster_id = on_schedule["cluster_id"]

    if cluster_id in off_schedules:

        for off_schedule in off_schedules[cluster_id]:

            duration = int(off_schedule["mean_duration"])

            all_data.append(f"{hh}:{mm:02d},{duration}")

payload = ";".join(all_data)

WEBAPP_URL = "https://script.google.com/macros/s/AKfycbxH3z5lQzbnlx0MYkIWoK2o1ral6nerUyp0SM1GYTY6l2bs6ZPaMFxasT_lhVbLQ4ZPOQ/exec"

r = requests.get(
    WEBAPP_URL,
    params={"data": payload}
)

print(payload)
print(r.text)
