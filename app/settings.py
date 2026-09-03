"""User-tunable settings for the scoreboard app.

Everything that depends on the machine or on personal taste lives here:
file names, timings, overlay geometry and window size.
"""

from __future__ import annotations

PROTOCOL = 2
FILE_SET = "2026-09-06-a"

# --- files ---------------------------------------------------------------
CONFIG_PATH = "config.json"      # slot boxes and reading thresholds
MODEL_PATH = "digits.npz"        # trained weights used to read digits
OUTPUT_PATH = "scores.json"      # standings dumped for other tools

# --- capture -------------------------------------------------------------
DEFAULT_INTERVAL = 0.5           # seconds between reads
FIRST_FRAME_TIMEOUT = 8.0        # give up if no frame arrives within this time
WGC_WAIT = 1.0                   # fall back to screen grabbing after this wait

# --- reading -------------------------------------------------------------
MODEL_MIN_SCORE = 0.60           # accept a digit only above this probability
MODEL_MIN_MARGIN = 0.20          # ...and only if it beats the runner-up by this
INCOMPLETE_SCORE = 0.85          # stricter bar while few digits are known

# --- overlay -------------------------------------------------------------
OVERLAY_PORT = 8777              # first port to try; taken ports are skipped
OVERLAY_PORT_TRIES = 20
OVERLAY_TITLE = "LEADERBOARD"
OVERLAY_WIDTH = 207              # box width for the OBS browser source
OVERLAY_HEIGHT = 242             # height to set in OBS
OVERLAY_ROW_HEIGHT = 25
MOVE_MS = 770                    # row travel time when the order changes
HOLD_MS = 140                    # extra time the highlight stays on

# --- window --------------------------------------------------------------
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
LOG_HEIGHT = 170                 # extra height while the log is open
PREVIEW_SCALE = 1.3              # preview magnification inside the window

# --- timeline correction -------------------------------------------------
TIMELINE_ON = True               # start with timeline correction enabled
SMALL_WINDOW = 9                 # frames kept for spike rollback
BIG_WINDOW = 40                  # value changes kept for trend checks
REVERT_GAP = 1                   # cost gap required to drop the current value
