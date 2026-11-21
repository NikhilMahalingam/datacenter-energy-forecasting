from io import BytesIO
from urllib.error import HTTPError

import pytest

from esif_forecasting import download


class Response(BytesIO):
    headers = {}


def test_failed_responses_leave_no_partial_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: Response(b"<html>"))
    path = tmp_path / "power.parquet"
    with pytest.raises(ValueError, match="file format"):
        download.retrieve("https://example.org/power.parquet", path)
    assert list(tmp_path.iterdir()) == []


def test_retries_then_uses_zip_and_records_source(tmp_path, monkeypatch):
    attempts = []

    def fetch(request, **kwargs):
        attempts.append(request.full_url)
        if request.full_url.endswith(".parquet"):
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)
        return Response(b"PKtest")

    monkeypatch.setattr(download, "urlopen", fetch)
    monkeypatch.setattr(download, "sleep", lambda _: None)
    download.download_file("power", tmp_path)
    assert len(attempts) == 5
    name = download.FILES["power"][1]
    assert (tmp_path / name).read_bytes() == b"PKtest"
    assert attempts[-1] in (tmp_path / (name + ".json")).read_text()
    download.download_file("power", tmp_path)
    assert len(attempts) == 5
