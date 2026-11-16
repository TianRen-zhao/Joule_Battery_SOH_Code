import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get('CRPS_BASE_PATH', Path(__file__).resolve().parent / 'workspace'))
STAGE4_RUN = Path(os.environ['CRPS_STAGE4_RUN']) if os.environ.get('CRPS_STAGE4_RUN') else None
OUTPUT = Path(os.environ.get('CRPS_REPEAT_OUTPUT', DATA_ROOT / 'JOULE_SOURCE_REPEATS_V2'))
