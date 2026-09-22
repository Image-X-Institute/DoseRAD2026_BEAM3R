"""Run-directory naming.

``generate_unique_log_directory`` calls ``os.path.relpath``, so the run directory
is created relative to the *current working directory* at launch. Do not ``cd``
during a run.
"""

import datetime
import os


def generate_datetime() -> str:
    return f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S%z}"


def sanitise_output_directory(output_directory: str) -> str:
    return os.path.relpath(output_directory)


def generate_unique_log_directory(output_directory: str) -> str:
    output_directory = sanitise_output_directory(output_directory)
    datetime_ = generate_datetime()
    log_path = os.path.join(output_directory, datetime_)
    os.makedirs(log_path, exist_ok=True)
    return log_path
