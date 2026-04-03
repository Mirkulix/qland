"""
Flughafen-Datenbank: Alle relevanten Flughäfen in DE, FR, NL
mit IATA-Code, Name, Land, Koordinaten
"""

AIRPORTS = [
    # === DEUTSCHLAND ===
    {"iata": "HAM", "name": "Hamburg", "country": "DE", "lat": 53.630, "lon": 9.988},
    {"iata": "FRA", "name": "Frankfurt", "country": "DE", "lat": 50.033, "lon": 8.570},
    {"iata": "MUC", "name": "München", "country": "DE", "lat": 48.354, "lon": 11.786},
    {"iata": "DUS", "name": "Düsseldorf", "country": "DE", "lat": 51.289, "lon": 6.767},
    {"iata": "CGN", "name": "Köln/Bonn", "country": "DE", "lat": 50.866, "lon": 7.143},
    {"iata": "BER", "name": "Berlin Brandenburg", "country": "DE", "lat": 52.362, "lon": 13.390},
    {"iata": "STR", "name": "Stuttgart", "country": "DE", "lat": 48.690, "lon": 9.222},
    {"iata": "HAJ", "name": "Hannover", "country": "DE", "lat": 52.461, "lon": 9.685},
    {"iata": "NUE", "name": "Nürnberg", "country": "DE", "lat": 49.499, "lon": 11.067},
    {"iata": "BRE", "name": "Bremen", "country": "DE", "lat": 53.047, "lon": 8.787},
    {"iata": "DTM", "name": "Dortmund", "country": "DE", "lat": 51.518, "lon": 7.612},
    {"iata": "FMO", "name": "Münster/Osnabrück", "country": "DE", "lat": 52.135, "lon": 7.685},
    {"iata": "PAD", "name": "Paderborn", "country": "DE", "lat": 51.614, "lon": 8.616},
    {"iata": "NRN", "name": "Weeze (Niederrhein)", "country": "DE", "lat": 51.602, "lon": 6.142},
    {"iata": "FDH", "name": "Friedrichshafen", "country": "DE", "lat": 47.671, "lon": 9.511},
    {"iata": "FKB", "name": "Karlsruhe/Baden-Baden", "country": "DE", "lat": 48.779, "lon": 8.080},
    {"iata": "HHN", "name": "Frankfurt-Hahn", "country": "DE", "lat": 49.949, "lon": 7.264},
    {"iata": "MMX", "name": "Memmingen (Allgäu)", "country": "DE", "lat": 47.988, "lon": 10.239},
    {"iata": "LEJ", "name": "Leipzig/Halle", "country": "DE", "lat": 51.424, "lon": 12.236},
    {"iata": "DRS", "name": "Dresden", "country": "DE", "lat": 51.133, "lon": 13.767},
    {"iata": "ERF", "name": "Erfurt-Weimar", "country": "DE", "lat": 50.980, "lon": 10.958},
    {"iata": "SCN", "name": "Saarbrücken", "country": "DE", "lat": 49.215, "lon": 7.109},
    {"iata": "LBC", "name": "Lübeck", "country": "DE", "lat": 53.805, "lon": 10.719},
    {"iata": "RLG", "name": "Rostock-Laage", "country": "DE", "lat": 53.918, "lon": 12.278},

    # === FRANKREICH ===
    {"iata": "CDG", "name": "Paris Charles de Gaulle", "country": "FR", "lat": 49.010, "lon": 2.548},
    {"iata": "ORY", "name": "Paris Orly", "country": "FR", "lat": 48.726, "lon": 2.359},
    {"iata": "BVA", "name": "Paris Beauvais", "country": "FR", "lat": 49.454, "lon": 2.113},
    {"iata": "LYS", "name": "Lyon", "country": "FR", "lat": 45.726, "lon": 5.091},
    {"iata": "MRS", "name": "Marseille", "country": "FR", "lat": 43.436, "lon": 5.215},
    {"iata": "TLS", "name": "Toulouse", "country": "FR", "lat": 43.629, "lon": 1.364},
    {"iata": "NTE", "name": "Nantes", "country": "FR", "lat": 47.153, "lon": -1.611},
    {"iata": "BOD", "name": "Bordeaux", "country": "FR", "lat": 44.828, "lon": -0.715},
    {"iata": "SXB", "name": "Straßburg", "country": "FR", "lat": 48.538, "lon": 7.628},
    {"iata": "LIL", "name": "Lille", "country": "FR", "lat": 50.563, "lon": 3.087},
    {"iata": "MPL", "name": "Montpellier", "country": "FR", "lat": 43.576, "lon": 3.963},
    {"iata": "NCE", "name": "Nizza", "country": "FR", "lat": 43.658, "lon": 7.216},

    # === NIEDERLANDE ===
    {"iata": "AMS", "name": "Amsterdam Schiphol", "country": "NL", "lat": 52.309, "lon": 4.764},
    {"iata": "EIN", "name": "Eindhoven", "country": "NL", "lat": 51.450, "lon": 5.374},
    {"iata": "RTM", "name": "Rotterdam Den Haag", "country": "NL", "lat": 51.957, "lon": 4.438},
    {"iata": "GRQ", "name": "Groningen Eelde", "country": "NL", "lat": 53.120, "lon": 6.579},
    {"iata": "MST", "name": "Maastricht Aachen", "country": "NL", "lat": 50.912, "lon": 5.770},

    # === BELGIEN (Bonus - nah an DE/NL) ===
    {"iata": "BRU", "name": "Brüssel", "country": "BE", "lat": 50.901, "lon": 4.484},
    {"iata": "CRL", "name": "Brüssel-Charleroi", "country": "BE", "lat": 50.459, "lon": 4.453},

    # === LUXEMBURG ===
    {"iata": "LUX", "name": "Luxemburg", "country": "LU", "lat": 49.627, "lon": 6.212},
]

# Hamburg Koordinaten als Startpunkt
HAMBURG = {"lat": 53.551, "lon": 9.994}
