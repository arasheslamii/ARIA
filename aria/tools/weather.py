"""Weather via Open-Meteo — free, no API key, portable.

Resolves a location (named city → auto-detected current location → configured
home → ask), then fetches current conditions + a short daily forecast and returns
a warm, spoken-friendly summary.

Auto-location uses IP-based geolocation: it asks a free third-party service for the
approximate city/lat-lon of the machine's PUBLIC IP (no GPS, city-level accuracy),
cached briefly. Privacy: this sends your public IP to that service only when you
ask for the weather without naming a city.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from aria.tools.base import Tool, ToolError, ToolResult

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_IPLOC_URL = "https://ipapi.co/json/"  # free, no key, HTTPS; city-level from public IP
_IPLOC_TTL_S = 600.0  # cache the detected location for 10 min

# WMO weather interpretation codes -> (short description, is precipitation).
_WMO: dict[int, tuple[str, bool]] = {
    0: ("clear", False),
    1: ("mostly clear", False),
    2: ("partly cloudy", False),
    3: ("cloudy", False),
    45: ("foggy", False),
    48: ("foggy", False),
    51: ("drizzly", True),
    53: ("drizzly", True),
    55: ("drizzly", True),
    56: ("freezing drizzle", True),
    57: ("freezing drizzle", True),
    61: ("light rain", True),
    63: ("rainy", True),
    65: ("heavy rain", True),
    66: ("freezing rain", True),
    67: ("freezing rain", True),
    71: ("light snow", True),
    73: ("snowy", True),
    75: ("heavy snow", True),
    77: ("snow grains", True),
    80: ("rain showers", True),
    81: ("rain showers", True),
    82: ("heavy showers", True),
    85: ("snow showers", True),
    86: ("snow showers", True),
    95: ("thunderstorms", True),
    96: ("thunderstorms", True),
    99: ("thunderstorms with hail", True),
}


def _describe(code: int) -> tuple[str, bool]:
    return _WMO.get(int(code), ("mixed", False))


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get the current weather and today's forecast for a city. Use for 'what's "
        "the weather'. If no city is given, uses the user's home location."
    )
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name; optional (defaults to home).",
            }
        },
    }
    risk = "safe"

    def __init__(self, home_location: str | None = None, timeout: float = 8.0) -> None:
        self._home = home_location
        self._timeout = timeout
        self._ip_cache: tuple[float, dict] | None = None

    async def run(self, **kwargs: Any) -> ToolResult:
        explicit = str(kwargs.get("location") or "").strip()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            place = await self._resolve_place(client, explicit)
            forecast = await self._forecast(client, place["latitude"], place["longitude"])

        cur = forecast["current"]
        daily = forecast["daily"]
        temp = round(cur["temperature_2m"])
        desc, raining_now = _describe(cur["weather_code"])
        high = round(daily["temperature_2m_max"][0])
        low = round(daily["temperature_2m_min"][0])
        precip = daily.get("precipitation_probability_max", [0])[0] or 0

        name = place["name"]
        spoken = f"It's {temp}° and {desc} in {name}"
        if precip >= 50 and not raining_now:
            spoken += ", rain likely later"
        spoken += f" — high of {high}, low of {low}."

        return ToolResult(
            content=(
                f"{name}: now {temp}° {desc}; today high {high}° / low {low}°; "
                f"precip chance {precip}%."
            ),
            data={"location": name, "temp": temp, "high": high, "low": low, "precip": precip},
            spoken=spoken,
        )

    async def _resolve_place(self, client: httpx.AsyncClient, explicit: str) -> dict:
        """Location priority: (1) named city, (2) auto-detected current location,
        (3) configured home, (4) ask."""
        if explicit:
            return await self._geocode(client, explicit)
        detected = await self._iploc(client)
        if detected is not None:
            return detected
        if self._home:
            return await self._geocode(client, self._home)
        raise ToolError(
            "Which city's weather would you like? I couldn't detect your location."
        )

    async def _iploc(self, client: httpx.AsyncClient) -> dict | None:
        """Approximate current location from the public IP (cached). None on failure."""
        now = time.monotonic()
        if self._ip_cache and now - self._ip_cache[0] < _IPLOC_TTL_S:
            return self._ip_cache[1]
        try:
            r = await client.get(_IPLOC_URL)
            r.raise_for_status()
            data = r.json()
            lat, lon = data.get("latitude"), data.get("longitude")
            if lat is None or lon is None:
                return None
            place = {
                "name": data.get("city") or "your area",
                "latitude": lat,
                "longitude": lon,
            }
            self._ip_cache = (now, place)
            return place
        except (httpx.HTTPError, ValueError):
            return None

    async def _geocode(self, client: httpx.AsyncClient, location: str) -> dict:
        try:
            r = await client.get(
                _GEOCODE_URL, params={"name": location, "count": 1, "language": "en"}
            )
            r.raise_for_status()
            results = r.json().get("results") or []
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"I couldn't look up that location ({exc}).") from exc
        if not results:
            raise ToolError(f"I couldn't find a place called {location}.")
        return results[0]

    async def _forecast(self, client: httpx.AsyncClient, lat: float, lon: float) -> dict:
        try:
            r = await client.get(
                _FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": 1,
                },
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"I couldn't get the forecast ({exc}).") from exc
