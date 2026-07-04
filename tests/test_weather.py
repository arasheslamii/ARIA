"""Weather tool (Open-Meteo, no API key) — Milestone 4 Part A."""

from __future__ import annotations

import httpx
import pytest

from aria.tools.base import ToolError
from aria.tools.weather import WeatherTool

_GEO = {"results": [{"name": "Edinburgh", "latitude": 55.95, "longitude": -3.19, "country": "UK"}]}
_FC = {
    "current": {"temperature_2m": 13.6, "weather_code": 3},  # cloudy
    "daily": {
        "weather_code": [61],
        "temperature_2m_max": [17.2],
        "temperature_2m_min": [9.1],
        "precipitation_probability_max": [70],
    },
}


def _patch(monkeypatch, geo=_GEO, fc=_FC, ip=None, seen=None):
    async def fake_get(self, url, **kwargs):
        u = str(url)
        if seen is not None:
            seen.append(u)
        if "ipapi" in u:
            if ip == "boom":
                raise httpx.ConnectError("no network")
            return httpx.Response(200, json=(ip or {}), request=httpx.Request("GET", u))
        body = geo if "geocoding" in u else fc
        return httpx.Response(200, json=body, request=httpx.Request("GET", u))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


async def test_weather_spoken_summary(monkeypatch):
    _patch(monkeypatch)
    res = await WeatherTool().run(location="Edinburgh")
    assert res.spoken == (
        "It's 14° and cloudy in Edinburgh, rain likely later — high of 17, low of 9."
    )
    assert res.data["temp"] == 14 and res.data["high"] == 17


async def test_weather_uses_home_location_when_none_given(monkeypatch):
    _patch(monkeypatch)
    res = await WeatherTool(home_location="Edinburgh").run()  # no location arg
    assert "Edinburgh" in res.spoken


async def test_weather_no_location_no_home_asks(monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(ToolError, match="[Ww]hich city"):
        await WeatherTool().run()  # no arg, no home


async def test_weather_unknown_city_honest_error(monkeypatch):
    _patch(monkeypatch, geo={"results": []})  # geocode finds nothing
    with pytest.raises(ToolError, match="couldn't find"):
        await WeatherTool().run(location="Nowheresville")


async def test_weather_no_rain_clause_when_dry(monkeypatch):
    dry = {**_FC, "daily": {**_FC["daily"], "precipitation_probability_max": [10]}}
    _patch(monkeypatch, fc=dry)
    res = await WeatherTool().run(location="Edinburgh")
    assert "rain likely" not in res.spoken


def test_config_has_home_location():
    from aria.config.schema import AriaConfig

    assert AriaConfig().home_location is None  # default unset


# --- auto-detect current location (IP geolocation) ------------------------
async def test_weather_auto_detects_current_location(monkeypatch):
    _patch(monkeypatch, ip={"city": "Berlin", "latitude": 52.5, "longitude": 13.4})
    res = await WeatherTool().run()  # no city, no home -> auto-detect
    assert "Berlin" in res.spoken


async def test_explicit_city_overrides_autodetect(monkeypatch):
    seen: list[str] = []
    _patch(monkeypatch, ip={"city": "Berlin", "latitude": 52.5, "longitude": 13.4}, seen=seen)
    res = await WeatherTool().run(location="Paris")  # explicit wins
    assert "Edinburgh" in res.spoken  # (geocode mock returns Edinburgh)
    assert not any("ipapi" in u for u in seen)  # IP geolocation NOT consulted


async def test_geolocation_failure_falls_back_to_home(monkeypatch):
    _patch(monkeypatch, ip="boom")  # IP geolocation unreachable
    res = await WeatherTool(home_location="Edinburgh").run()  # no city
    assert "Edinburgh" in res.spoken  # fell back to configured home


async def test_detected_location_is_cached(monkeypatch):
    seen: list[str] = []
    _patch(monkeypatch, ip={"city": "Berlin", "latitude": 52.5, "longitude": 13.4}, seen=seen)
    tool = WeatherTool()
    await tool.run()
    await tool.run()  # second ask within TTL
    assert sum(1 for u in seen if "ipapi" in u) == 1  # geolocation fetched once, cached
