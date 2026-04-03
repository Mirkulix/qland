"""
Travel Search - Günstigste Flüge nach Alicante finden
Web-App mit Apify Google Flights Scraper
"""

import os
import json
import math
import time
import threading
from datetime import datetime

from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ============================================================
# Globaler Status für die Suche (damit UI live Updates bekommt)
# ============================================================
search_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_airport": "",
    "results": [],
    "errors": [],
    "done": False,
}


# ============================================================
# Hilfsfunktionen
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Entfernung zwischen zwei Koordinaten in km (Luftlinie)."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def estimate_train_cost(distance_km):
    """
    Schätzt Zugkosten Hamburg → Flughafen basierend auf Entfernung.
    Basierend auf DB Sparpreis-Erfahrungswerten.
    """
    if distance_km < 50:
        return 15.0   # Nahverkehr
    elif distance_km < 150:
        return 25.0   # Regionaler Sparpreis
    elif distance_km < 300:
        return 35.0   # ICE Sparpreis kurz
    elif distance_km < 500:
        return 50.0   # ICE Sparpreis mittel
    elif distance_km < 800:
        return 65.0   # ICE Sparpreis lang
    elif distance_km < 1200:
        return 80.0   # International (z.B. Paris, Amsterdam)
    else:
        return 100.0  # Weit entfernt


def estimate_train_duration_hours(distance_km):
    """Schätzt Zugdauer basierend auf Entfernung."""
    if distance_km < 100:
        return round(distance_km / 80, 1)   # Nahverkehr ~80 km/h
    elif distance_km < 500:
        return round(distance_km / 120, 1)  # ICE ~120 km/h effektiv
    else:
        return round(distance_km / 100, 1)  # International ~100 km/h


# ============================================================
# Apify Google Flights Scraper
# ============================================================

def search_flights_apify(origin_iata, destination_iata, date_str, adults=2, children=1):
    """
    Sucht Flüge über Apify Google Flights Scraper.

    Args:
        origin_iata: Abflughafen IATA-Code (z.B. "HAM")
        destination_iata: Zielflughafen IATA-Code (z.B. "ALC")
        date_str: Datum im Format "YYYY-MM-DD"
        adults: Anzahl Erwachsene
        children: Anzahl Kinder

    Returns:
        Liste von Flug-Ergebnissen
    """
    try:
        from apify_client import ApifyClient
    except ImportError:
        return _search_flights_apify_rest(origin_iata, destination_iata, date_str, adults, children)

    token = os.getenv("APIFY_TOKEN", "")
    if not token:
        raise ValueError("APIFY_TOKEN nicht gesetzt! Bitte in .env eintragen.")

    client = ApifyClient(token)

    run_input = {
        "departureAirportCode": origin_iata,
        "arrivalAirportCode": destination_iata,
        "departureDate": date_str,
        "returnDate": "",
        "adults": adults,
        "children": children,
        "infants": 0,
        "currency": "EUR",
        "market": "DE",
        "language": "de",
        "cabinClass": "economy",
        "maxStops": "any",
        "bags": 0,          # nur Handgepäck
    }

    # Google Flights Scraper Actor ID
    # Versuche verschiedene bekannte Actor IDs
    actor_ids = [
        "voyager/google-flights-scraper",
        "misceres/google-flights-scraper",
        "apify/google-flights-scraper",
    ]

    last_error = None
    for actor_id in actor_ids:
        try:
            run = client.actor(actor_id).call(run_input=run_input, timeout_secs=120)
            results = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                results.append(item)
            return results
        except Exception as e:
            last_error = e
            continue

    raise last_error or Exception("Kein passender Apify Actor gefunden")


def _search_flights_apify_rest(origin_iata, destination_iata, date_str, adults, children):
    """Fallback: Apify REST API direkt nutzen (ohne apify-client Paket)."""
    import requests

    token = os.getenv("APIFY_TOKEN", "")
    if not token:
        raise ValueError("APIFY_TOKEN nicht gesetzt!")

    # Versuche den Actor über die REST API
    actor_ids = [
        "voyager~google-flights-scraper",
        "misceres~google-flights-scraper",
        "apify~google-flights-scraper",
    ]

    run_input = {
        "departureAirportCode": origin_iata,
        "arrivalAirportCode": destination_iata,
        "departureDate": date_str,
        "returnDate": "",
        "adults": adults,
        "children": children,
        "infants": 0,
        "currency": "EUR",
        "market": "DE",
        "language": "de",
        "cabinClass": "economy",
        "maxStops": "any",
        "bags": 0,
    }

    last_error = None
    for actor_id in actor_ids:
        try:
            url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
            headers = {"Content-Type": "application/json"}
            params = {"token": token, "timeout": 120, "waitForFinish": 120}

            resp = requests.post(url, json=run_input, headers=headers, params=params, timeout=180)

            if resp.status_code == 404:
                continue

            resp.raise_for_status()
            run_data = resp.json().get("data", {})
            dataset_id = run_data.get("defaultDatasetId")

            if dataset_id:
                ds_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
                ds_resp = requests.get(ds_url, params={"token": token}, timeout=60)
                ds_resp.raise_for_status()
                return ds_resp.json()

            return []
        except Exception as e:
            last_error = e
            continue

    raise last_error or Exception("Kein Apify Actor gefunden")


# ============================================================
# DEMO-Modus (ohne API Key zum Testen der UI)
# ============================================================

def search_flights_demo(origin_iata, destination_iata, date_str, adults=2, children=1):
    """
    Demo-Daten für UI-Test ohne API Key.
    Simuliert realistische Preise basierend auf bekannten Routen.
    """
    import random

    known_direct_routes = {
        "HAM": {"price_range": (75, 180), "airlines": ["Ryanair", "Eurowings"], "direct": True, "duration": "3h 10m"},
        "DUS": {"price_range": (60, 150), "airlines": ["Eurowings", "Ryanair"], "direct": True, "duration": "2h 50m"},
        "CGN": {"price_range": (55, 140), "airlines": ["Eurowings", "Ryanair"], "direct": True, "duration": "2h 45m"},
        "FRA": {"price_range": (70, 200), "airlines": ["Lufthansa", "Ryanair", "Condor"], "direct": True, "duration": "2h 40m"},
        "BER": {"price_range": (65, 170), "airlines": ["Ryanair", "easyJet"], "direct": True, "duration": "3h 15m"},
        "STR": {"price_range": (55, 130), "airlines": ["Eurowings", "Ryanair"], "direct": True, "duration": "2h 20m"},
        "MUC": {"price_range": (70, 190), "airlines": ["Lufthansa", "Vueling", "Eurowings"], "direct": True, "duration": "2h 25m"},
        "NUE": {"price_range": (50, 120), "airlines": ["Ryanair"], "direct": True, "duration": "2h 35m"},
        "FDH": {"price_range": (35, 90), "airlines": ["Ryanair"], "direct": True, "duration": "2h 15m"},
        "NRN": {"price_range": (30, 85), "airlines": ["Ryanair"], "direct": True, "duration": "2h 40m"},
        "HHN": {"price_range": (35, 95), "airlines": ["Ryanair"], "direct": True, "duration": "2h 30m"},
        "MMX": {"price_range": (30, 80), "airlines": ["Ryanair"], "direct": True, "duration": "2h 10m"},
        "FKB": {"price_range": (35, 90), "airlines": ["Ryanair"], "direct": True, "duration": "2h 20m"},
        "BRE": {"price_range": (60, 140), "airlines": ["Ryanair"], "direct": True, "duration": "3h 00m"},
        "DTM": {"price_range": (45, 110), "airlines": ["Eurowings"], "direct": True, "duration": "2h 50m"},
        "LEJ": {"price_range": (55, 130), "airlines": ["Ryanair"], "direct": True, "duration": "3h 00m"},
        "CDG": {"price_range": (45, 130), "airlines": ["Vueling", "easyJet", "Transavia"], "direct": True, "duration": "2h 15m"},
        "ORY": {"price_range": (40, 120), "airlines": ["Vueling", "Transavia"], "direct": True, "duration": "2h 10m"},
        "BVA": {"price_range": (30, 80), "airlines": ["Ryanair"], "direct": True, "duration": "2h 20m"},
        "LYS": {"price_range": (35, 100), "airlines": ["Volotea", "easyJet"], "direct": True, "duration": "1h 40m"},
        "MRS": {"price_range": (30, 90), "airlines": ["Ryanair", "Volotea"], "direct": True, "duration": "1h 20m"},
        "TLS": {"price_range": (35, 95), "airlines": ["Ryanair", "Volotea"], "direct": True, "duration": "1h 25m"},
        "BOD": {"price_range": (40, 100), "airlines": ["Ryanair", "Volotea"], "direct": True, "duration": "1h 35m"},
        "AMS": {"price_range": (55, 150), "airlines": ["Transavia", "KLM", "Vueling"], "direct": True, "duration": "2h 35m"},
        "EIN": {"price_range": (35, 90), "airlines": ["Ryanair", "Transavia"], "direct": True, "duration": "2h 25m"},
        "BRU": {"price_range": (45, 120), "airlines": ["Ryanair", "Vueling"], "direct": True, "duration": "2h 30m"},
        "CRL": {"price_range": (25, 75), "airlines": ["Ryanair"], "direct": True, "duration": "2h 35m"},
        "NCE": {"price_range": (40, 110), "airlines": ["easyJet", "Volotea"], "direct": True, "duration": "1h 15m"},
        "MPL": {"price_range": (35, 95), "airlines": ["Ryanair"], "direct": True, "duration": "1h 20m"},
        "LIL": {"price_range": (50, 130), "airlines": ["Volotea"], "direct": False, "duration": "4h 30m"},
        "NTE": {"price_range": (40, 110), "airlines": ["Volotea", "easyJet"], "direct": True, "duration": "1h 55m"},
    }

    route = known_direct_routes.get(origin_iata)
    if not route:
        # Generische Preise für unbekannte Flughäfen
        base_price = random.randint(80, 200)
        return [{
            "airline": "Diverse",
            "price_per_person": base_price,
            "price_total": base_price * (adults + children),
            "direct": False,
            "duration": "4h+",
            "departure_time": "variabel",
            "stops": 1,
            "note": "Umsteigeverbindung",
        }]

    results = []
    for airline in route["airlines"]:
        low, high = route["price_range"]
        price_pp = random.randint(low, high)
        # Juli = Hochsaison, Aufschlag
        price_pp = int(price_pp * 1.2)
        # Kind oft gleicher Preis bei Low-Cost
        total = price_pp * adults + int(price_pp * 0.9) * children

        dep_hours = random.choice(["06:15", "07:30", "09:45", "11:20", "14:05", "16:30", "18:45", "21:10"])
        results.append({
            "airline": airline,
            "price_per_person": price_pp,
            "price_total": total,
            "direct": route["direct"],
            "duration": route["duration"],
            "departure_time": dep_hours,
            "stops": 0 if route["direct"] else 1,
            "note": "Nur Handgepäck 10kg" if airline in ["Ryanair", "easyJet", "Wizz Air"] else "Handgepäck inkl.",
        })

    return sorted(results, key=lambda x: x["price_total"])


# ============================================================
# Such-Thread (läuft im Hintergrund)
# ============================================================

def run_search(params):
    """Führt die komplette Suche durch alle Flughäfen durch."""
    global search_status

    from airports import AIRPORTS, HAMBURG

    date_str = params.get("date", "2026-07-22")
    adults = int(params.get("adults", 2))
    children = int(params.get("children", 1))
    max_distance = int(params.get("max_distance", 1500))
    use_demo = params.get("demo", "false") == "true"
    destination = params.get("destination", "ALC")

    # Filter Flughäfen nach Entfernung
    candidates = []
    for ap in AIRPORTS:
        dist = haversine_km(HAMBURG["lat"], HAMBURG["lon"], ap["lat"], ap["lon"])
        if dist <= max_distance:
            ap_copy = dict(ap)
            ap_copy["distance_from_hamburg"] = round(dist, 1)
            ap_copy["train_cost_pp"] = estimate_train_cost(dist)
            ap_copy["train_duration"] = estimate_train_duration_hours(dist)
            candidates.append(ap_copy)

    # Sortiere nach Entfernung
    candidates.sort(key=lambda x: x["distance_from_hamburg"])

    search_status["total"] = len(candidates)
    search_status["progress"] = 0
    search_status["results"] = []
    search_status["errors"] = []

    for i, airport in enumerate(candidates):
        search_status["progress"] = i + 1
        search_status["current_airport"] = f"{airport['name']} ({airport['iata']})"

        try:
            if use_demo:
                flights = search_flights_demo(airport["iata"], destination, date_str, adults, children)
            else:
                flights = search_flights_apify(airport["iata"], destination, date_str, adults, children)

            if not flights:
                continue

            for flight in flights:
                total_persons = adults + children

                # Anreisekosten für alle Personen
                train_total = airport["train_cost_pp"] * total_persons
                # Hamburg selbst = 0 Anreise
                if airport["iata"] == "HAM":
                    train_total = 0
                    airport["train_cost_pp"] = 0
                    airport["train_duration"] = 0

                flight_total = flight.get("price_total", flight.get("price_per_person", 0) * total_persons)

                result = {
                    "airport_iata": airport["iata"],
                    "airport_name": airport["name"],
                    "country": airport["country"],
                    "distance_km": airport["distance_from_hamburg"],
                    "train_cost_pp": airport["train_cost_pp"],
                    "train_cost_total": train_total,
                    "train_duration_h": airport["train_duration"],
                    "airline": flight.get("airline", "Unbekannt"),
                    "flight_price_pp": flight.get("price_per_person", 0),
                    "flight_price_total": flight_total,
                    "total_cost": round(train_total + flight_total, 2),
                    "direct_flight": flight.get("direct", False),
                    "flight_duration": flight.get("duration", "?"),
                    "departure_time": flight.get("departure_time", "?"),
                    "stops": flight.get("stops", 0),
                    "note": flight.get("note", ""),
                }
                search_status["results"].append(result)

        except Exception as e:
            search_status["errors"].append({
                "airport": airport["iata"],
                "error": str(e),
            })

        # Sortiere bisherige Ergebnisse nach Gesamtpreis
        search_status["results"].sort(key=lambda x: x["total_cost"])

        # Kleine Pause zwischen API-Calls (nicht im Demo-Modus)
        if not use_demo:
            time.sleep(2)

    search_status["running"] = False
    search_status["done"] = True


# ============================================================
# Flask Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def start_search():
    """Startet die Flugsuche."""
    global search_status

    if search_status["running"]:
        return jsonify({"error": "Suche läuft bereits!"}), 409

    search_status = {
        "running": True,
        "progress": 0,
        "total": 0,
        "current_airport": "",
        "results": [],
        "errors": [],
        "done": False,
    }

    params = request.json or {}
    thread = threading.Thread(target=run_search, args=(params,), daemon=True)
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/status")
def get_status():
    """Gibt den aktuellen Such-Status zurück."""
    return jsonify({
        "running": search_status["running"],
        "progress": search_status["progress"],
        "total": search_status["total"],
        "current_airport": search_status["current_airport"],
        "result_count": len(search_status["results"]),
        "results": search_status["results"][:50],  # Top 50
        "errors": search_status["errors"][-5:],     # Letzte 5 Fehler
        "done": search_status["done"],
    })


@app.route("/api/results")
def get_results():
    """Gibt alle Ergebnisse zurück."""
    return jsonify({
        "results": search_status["results"],
        "errors": search_status["errors"],
        "done": search_status["done"],
    })


@app.route("/api/airports")
def get_airports():
    """Gibt alle Flughäfen mit Entfernungen zurück."""
    from airports import AIRPORTS, HAMBURG

    result = []
    for ap in AIRPORTS:
        dist = haversine_km(HAMBURG["lat"], HAMBURG["lon"], ap["lat"], ap["lon"])
        result.append({
            "iata": ap["iata"],
            "name": ap["name"],
            "country": ap["country"],
            "distance_km": round(dist, 1),
            "train_cost_pp": estimate_train_cost(dist),
            "train_duration_h": estimate_train_duration_hours(dist),
        })
    result.sort(key=lambda x: x["distance_km"])
    return jsonify(result)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TRAVEL SEARCH - Günstigste Flüge nach Alicante")
    print("  Öffne im Browser: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
