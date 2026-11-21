import json
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

CATALOG = "https://data.openei.org/submissions/8616"
ALTERNATE = "https://data.nlr.gov/submissions/300"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
BASE = "https://data.nrel.gov/system/files/300/"
FILES = {
    "power": (
        "1757103411-esif.influx.buildingData.PUE.combined.parquet",
        "1757105566-esif.influx.buildingData.PUE.combined.csv.zip",
    ),
    "weather": (
        "1757105566-esif.influx.buildingData.outside.combined_2.parquet",
        "1757105566-esif.influx.buildingData.outside.combined.csv.zip",
    ),
}


def retrieve(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    try:
        request = Request(url, headers={"User-Agent": "ESIF-forecasting/0.1"})
        with urlopen(request, timeout=30) as response, partial.open("wb") as output:
            magic = b"PAR1" if destination.suffix == ".parquet" else b"PK"
            header = response.read(len(magic))
            if header != magic:
                raise ValueError("response is not the advertised file format")
            output.write(header)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            size = response.headers.get("Content-Length")
            if size and output.tell() != int(size):
                raise ValueError("incomplete download")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_file(kind: str, directory: Path) -> None:
    if any((directory / name).is_file() for name in FILES[kind]):
        return
    for name in FILES[kind]:
        primary = BASE + name
        for url in (primary, primary.replace("data.nrel.gov", "data.nlr.gov")):
            for attempt in range(2):
                try:
                    retrieve(url, directory / name)
                except (OSError, URLError, ValueError) as error:
                    print(f"{url} (attempt {attempt + 1}): {error}", flush=True)
                    if attempt == 0:
                        sleep(1)
                else:
                    record = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
                    (directory / (name + ".json")).write_text(json.dumps(record), encoding="utf-8")
                    return
    raise FileNotFoundError(f"Download failed for {kind}. Open {CATALOG} or {ALTERNATE} "
                            f"and save its Parquet or zipped CSV in {directory}.")


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    failures = []
    for kind in FILES:
        try:
            download_file(kind, directory)
        except FileNotFoundError as error:
            failures.append(str(error))
    if failures:
        raise FileNotFoundError("\n".join(failures))
