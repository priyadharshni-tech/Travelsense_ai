import os
import streamlit as st
import streamlit.components.v1 as components
import json

def get_google_maps_api_key():
    """
    Retrieves Google Maps API Key from environment variables, Streamlit secrets, or session state.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        try:
            secrets_val = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
            if secrets_val:
                api_key = secrets_val
        except BaseException:
            pass
    if not api_key:
        try:
            session_val = st.session_state.get("user_google_maps_api_key", "")
            if session_val:
                api_key = session_val
        except BaseException:
            pass
    return str(api_key).strip() if api_key else ""


def render_google_maps_route_component(origin, destination, travel_mode="driving", api_key=""):
    """
    Renders an interactive HTML+JS Google Maps Route Planner component.
    Supports Places Autocomplete, Browser GPS Geolocation, Routes calculation,
    Alternative Route selection, interactive Markers & Polyline, fitted viewport,
    and Open in Google Maps external navigation.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
            * {{
                box-sizing: border-box;
                font-family: 'Plus Jakarta Sans', sans-serif;
            }}
            body {{
                margin: 0;
                padding: 10px;
                background-color: transparent;
                color: #0F172A;
            }}
            .route-card {{
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            }}
            .controls-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 12px;
                margin-bottom: 16px;
            }}
            .input-group {{
                display: flex;
                flex-direction: column;
            }}
            .input-group label {{
                font-size: 0.82rem;
                font-weight: 700;
                color: #334155;
                margin-bottom: 5px;
            }}
            .input-field {{
                padding: 10px 14px;
                border: 1px solid #CBD5E1;
                border-radius: 10px;
                font-size: 0.95rem;
                outline: none;
                transition: border-color 0.2s;
            }}
            .input-field:focus {{
                border-color: #0284C7;
                box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15);
            }}
            .btn {{
                padding: 10px 18px;
                border-radius: 10px;
                font-size: 0.9rem;
                font-weight: 700;
                cursor: pointer;
                border: none;
                transition: all 0.2s;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
            }}
            .btn-primary {{
                background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
                color: #FFFFFF;
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
            }}
            .btn-primary:hover {{
                background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
            }}
            .btn-secondary {{
                background: #F1F5F9;
                color: #0F172A;
                border: 1px solid #CBD5E1;
            }}
            .btn-secondary:hover {{
                background: #E2E8F0;
            }}
            .btn-success {{
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                color: #FFFFFF;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
                text-decoration: none;
            }}
            .btn-success:hover {{
                background: linear-gradient(135deg, #059669 0%, #047857 100%);
            }}
            #map {{
                width: 100%;
                height: 420px;
                border-radius: 14px;
                border: 1px solid #CBD5E1;
                margin-top: 14px;
                background-color: #F8FAFC;
            }}
            .stats-container {{
                display: flex;
                gap: 14px;
                flex-wrap: wrap;
                margin-top: 14px;
            }}
            .stat-badge {{
                flex: 1;
                min-width: 140px;
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 12px 16px;
                text-align: center;
            }}
            .stat-label {{
                font-size: 0.75rem;
                font-weight: 700;
                color: #64748B;
                text-transform: uppercase;
            }}
            .stat-value {{
                font-size: 1.3rem;
                font-weight: 800;
                color: #0284C7;
                margin-top: 2px;
            }}
            .alert-box {{
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 0.88rem;
                margin-top: 10px;
                display: none;
            }}
            .alert-warning {{
                background: #FFFBEB;
                border: 1px solid #FDE68A;
                color: #B45309;
            }}
            .alert-danger {{
                background: #FEF2F2;
                border: 1px solid #FCA5A5;
                color: #B91C1C;
            }}
            .alert-info {{
                background: #F0F9FF;
                border: 1px solid #BAE6FD;
                color: #0369A1;
            }}
            .routes-list {{
                margin-top: 14px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .route-option {{
                padding: 8px 14px;
                border-radius: 8px;
                border: 1px solid #CBD5E1;
                background: #FFFFFF;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;

            }}
            .route-option.active {{
                background: #0284C7;
                color: white;
                border-color: #0284C7;
            }}
        </style>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    </head>
    <body>
        <div class="route-card">
            <div class="controls-grid">
                <div class="input-group">
                    <label>📍 Starting Point (Origin)</label>
                    <div style="display: flex; gap: 6px;">
                        <input type="text" id="origin-input" class="input-field" style="flex:1;" value="{origin}" placeholder="Enter city, location, or GPS address...">
                        <button class="btn btn-secondary" onclick="useMyLocation()" title="Use Current GPS Location">📡 GPS</button>
                    </div>
                </div>

                <div class="input-group">
                    <label>🎯 Destination</label>
                    <input type="text" id="destination-input" class="input-field" value="{destination}" placeholder="Enter destination city...">
                </div>

                <div class="input-group">
                    <label>🚗 Travel Mode</label>
                    <select id="travel-mode" class="input-field" onchange="calculateRoute()">
                        <option value="DRIVING" {"selected" if travel_mode.upper()=="DRIVING" else ""}>🚗 Driving</option>
                        <option value="TRANSIT" {"selected" if travel_mode.upper()=="TRANSIT" else ""}>🚌 Transit</option>
                        <option value="WALKING" {"selected" if travel_mode.upper()=="WALKING" else ""}>🚶 Walking</option>
                        <option value="BICYCLING" {"selected" if travel_mode.upper()=="BICYCLING" else ""}>🚴 Cycling</option>
                    </select>
                </div>

                <div class="input-group" style="justify-content: flex-end;">
                    <label>&nbsp;</label>
                    <button class="btn btn-primary" onclick="calculateRoute()">🚀 Compute Smart Route</button>
                </div>
            </div>

            <div id="status-alert" class="alert-box"></div>

            <div class="stats-container">
                <div class="stat-badge">
                    <div class="stat-label">Total Distance</div>
                    <div class="stat-value" id="dist-val">-- km</div>
                </div>
                <div class="stat-badge">
                    <div class="stat-label">Est. Duration</div>
                    <div class="stat-value" id="dur-val">-- mins</div>
                </div>
                <div class="stat-badge">
                    <div class="stat-label">Traffic Mode</div>
                    <div class="stat-value" id="mode-val" style="font-size:1.05rem; color:#059669;">Normal</div>
                </div>
                <div class="stat-badge" style="display:flex; align-items:center; justify-content:center;">
                    <a id="gmaps-nav-btn" href="#" target="_blank" class="btn btn-success" style="width:100%; height:100%;">
                        🗺️ Open in Google Maps
                    </a>
                </div>
            </div>

            <div id="routes-container" class="routes-list" style="display:none;"></div>

            <div id="map"></div>
        </div>

        <script>
            let googleApiKey = "{api_key}";
            let map, directionsService, directionsRenderer;
            let leafletMap, leafletPolyline, leafletMarkers = [];
            let currentRoutes = [];

            function showAlert(msg, type='info') {{
                const el = document.getElementById('status-alert');
                el.className = 'alert-box alert-' + type;
                el.innerHTML = msg;
                el.style.display = 'block';
            }}

            function hideAlert() {{
                document.getElementById('status-alert').style.display = 'none';
            }}

            function updateNavButton(orig, dest, mode) {{
                const navBtn = document.getElementById('gmaps-nav-btn');
                const url = `https://www.google.com/maps/dir/?api=1&origin=${{encodeURIComponent(orig)}}&destination=${{encodeURIComponent(dest)}}&travelmode=${{mode.toLowerCase()}}`;
                navBtn.href = url;
            }}

            function useMyLocation() {{
                showAlert("📡 Requesting browser GPS permission...", "info");
                if (navigator.geolocation) {{
                    navigator.geolocation.getCurrentPosition(
                        (position) => {{
                            const lat = position.coords.latitude.toFixed(5);
                            const lng = position.coords.longitude.toFixed(5);
                            document.getElementById('origin-input').value = `${{lat}}, ${{lng}}`;
                            showAlert("✅ Current location updated from browser GPS!", "info");
                            calculateRoute();
                        }},
                        (error) => {{
                            showAlert("⚠️ GPS permission denied or location unavailable. Please enter starting city manually.", "warning");
                        }},
                        {{ timeout: 10000, enableHighAccuracy: true }}
                    );
                }} else {{
                    showAlert("⚠️ Geolocation API is not supported by your browser. Please enter location manually.", "warning");
                }}
            }}

            // Pre-populated coordinate database for instant fallback rendering if API key is missing
            const CITY_COORDS = {{
                "coimbatore": [11.0168, 76.9558],
                "ooty": [11.4102, 76.6950],
                "chennai": [13.0827, 80.2707],
                "bengaluru": [12.9716, 77.5946],
                "madurai": [9.9252, 78.1198],
                "kodaikanal": [10.2381, 77.4892],
                "jaipur": [26.9124, 75.7873],
                "udaipur": [24.5854, 73.7125],
                "munnar": [10.0889, 77.0595],
                "alleppey": [9.4981, 76.3388],
                "goa": [15.2993, 74.1240],
                "shimla": [31.1048, 77.1734],
                "manali": [32.2432, 77.1892],
                "kochi": [9.9312, 76.2673],
                "mysore": [12.2958, 76.6394],
                "agra": [27.1767, 78.0081],
                "varanasi": [25.3176, 82.9739],
                "rishikesh": [30.0869, 78.2676]
            }};

            function getApproxCoords(cityName) {{
                if (!cityName) return null;
                const clean = cityName.toLowerCase().split(',')[0].trim();
                for (let k in CITY_COORDS) {{
                    if (clean.includes(k) || k.includes(clean)) return CITY_COORDS[k];
                }}
                // Check lat,lng numeric string
                const parts = cityName.split(',');
                if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {{
                    return [parseFloat(parts[0]), parseFloat(parts[1])];
                }}
                return null;
            }}

            function renderLeafletFallback(orig, dest, mode) {{
                const oCoords = getApproxCoords(orig) || [11.0168, 76.9558];
                const dCoords = getApproxCoords(dest) || [11.4102, 76.6950];

                if (leafletMap) {{
                    leafletMap.remove();
                }}

                document.getElementById('map').innerHTML = '';
                leafletMap = L.map('map').setView(oCoords, 9);

                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; OpenStreetMap contributors'
                }}).addTo(leafletMap);

                const origMarker = L.marker(oCoords).addTo(leafletMap).bindPopup(`<b>Origin:</b> ${{orig}}`).openPopup();
                const destMarker = L.marker(dCoords).addTo(leafletMap).bindPopup(`<b>Destination:</b> ${{dest}}`);

                const latlngs = [oCoords, dCoords];
                leafletPolyline = L.polyline(latlngs, {{ color: '#0284C7', weight: 5, opacity: 0.8, dashArray: mode === 'TRANSIT' ? '10, 10' : null }}).addTo(leafletMap);
                leafletMap.fitBounds(L.latLngBounds(latlngs).pad(0.2));

                // Calculate approximate Haversine distance in km
                const R = 6371;
                const dLat = (dCoords[0] - oCoords[0]) * Math.PI / 180;
                const dLon = (dCoords[1] - oCoords[1]) * Math.PI / 180;
                const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                          Math.cos(oCoords[0] * Math.PI / 180) * Math.cos(dCoords[0] * Math.PI / 180) *
                          Math.sin(dLon/2) * Math.sin(dLon/2);
                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                let distKm = Math.round(R * c * 1.3); // add 30% winding factor
                if (distKm < 5) distKm = 85; // default fallback if same town coords

                let speedKmH = 45;
                if (mode === 'WALKING') speedKmH = 4.5;
                else if (mode === 'BICYCLING') speedKmH = 15;
                else if (mode === 'TRANSIT') speedKmH = 35;

                const durationHours = distKm / speedKmH;
                const hrs = Math.floor(durationHours);
                const mins = Math.round((durationHours - hrs) * 60);

                document.getElementById('dist-val').innerText = `${{distKm}} km`;
                document.getElementById('dur-val').innerText = hrs > 0 ? `${{hrs}}h ${{mins}}m` : `${{mins}} mins`;
                document.getElementById('mode-val').innerText = `${{mode}} (Est)`;

                showAlert(`ℹ️ Google Maps API Key not active in .env. Interactive map initialized with route bounds (${{distKm}} km).`, "info");
            }}

            function calculateGoogleRoute(orig, dest, mode) {{
                if (!directionsService) {{
                    directionsService = new google.maps.DirectionsService();
                    directionsRenderer = new google.maps.DirectionsRenderer({{
                        map: map,
                        panel: document.getElementById('routes-container')
                    }});
                }}

                const request = {{
                    origin: orig,
                    destination: dest,
                    travelMode: google.maps.TravelMode[mode],
                    provideRouteAlternatives: true,
                    drivingOptions: {{
                        departureTime: new Date(Date.now() + 5000),
                        trafficModel: 'bestguess'
                    }}
                }};

                directionsService.route(request, (result, status) => {{
                    if (status === 'OK') {{
                        hideAlert();
                        directionsRenderer.setDirections(result);
                        currentRoutes = result.routes;
                        displayRouteStats(result.routes[0], mode);
                        renderRouteSelector(result.routes);
                    }} else {{
                        showAlert(`⚠️ Could not compute route via Google Routes API (${{status}}). Falling back to visual route preview.`, "warning");
                        renderLeafletFallback(orig, dest, mode);
                    }}
                }});
            }}

            function displayRouteStats(route, mode) {{
                const leg = route.legs[0];
                const distText = leg.distance.text;
                const durText = leg.duration_in_traffic ? leg.duration_in_traffic.text + " (Live Traffic)" : leg.duration.text;

                document.getElementById('dist-val').innerText = distText;
                document.getElementById('dur-val').innerText = durText;
                document.getElementById('mode-val').innerText = mode;
            }}

            function renderRouteSelector(routes) {{
                const container = document.getElementById('routes-container');
                if (routes.length <= 1) {{
                    container.style.display = 'none';
                    return;
                }}
                container.style.display = 'flex';
                container.innerHTML = '<b style="align-self:center; font-size:0.85rem; color:#475569;">Alternative Routes:</b>';
                routes.forEach((rt, idx) => {{
                    const btn = document.createElement('button');
                    btn.className = `route-option ${{idx === 0 ? 'active' : ''}}`;
                    btn.innerText = `Route ${{idx + 1}}: ${{rt.summary || 'Via Main Road'}} (${{rt.legs[0].distance.text}}, ${{rt.legs[0].duration.text}})`;
                    btn.onclick = () => {{
                        document.querySelectorAll('.route-option').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        directionsRenderer.setRouteIndex(idx);
                        displayRouteStats(rt, document.getElementById('travel-mode').value);
                    }};
                    container.appendChild(btn);
                }});
            }}

            function calculateRoute() {{
                const orig = document.getElementById('origin-input').value.trim();
                const dest = document.getElementById('destination-input').value.trim();
                const mode = document.getElementById('travel-mode').value;

                if (!orig || !dest) {{
                    showAlert("⚠️ Please enter both starting location and destination.", "warning");
                    return;
                }}

                updateNavButton(orig, dest, mode);

                if (googleApiKey && window.google && window.google.maps) {{
                    calculateGoogleRoute(orig, dest, mode);
                }} else {{
                    renderLeafletFallback(orig, dest, mode);
                }}
            }}

            function initMap() {{
                if (googleApiKey && typeof google !== 'undefined' && google.maps) {{
                    const defaultCenter = {{ lat: 11.0168, lng: 76.9558 }};
                    map = new google.maps.Map(document.getElementById('map'), {{
                        zoom: 8,
                        center: defaultCenter,
                        mapTypeControl: true,
                        streetViewControl: false
                    }});
                    
                    // Attach Places Autocomplete if available
                    if (google.maps.places) {{
                        new google.maps.places.Autocomplete(document.getElementById('origin-input'));
                        new google.maps.places.Autocomplete(document.getElementById('destination-input'));
                    }}
                }}
                calculateRoute();
            }}

            window.onload = function() {{
                if (googleApiKey) {{
                    const script = document.createElement('script');
                    script.src = `https://maps.googleapis.com/maps/api/js?key=${{googleApiKey}}&libraries=places,routes&callback=initMap`;
                    script.async = true;
                    script.defer = true;
                    script.onerror = function() {{
                        showAlert("⚠️ Could not load Google Maps JS API script. Initializing visual interactive route engine.", "warning");
                        calculateRoute();
                    }};
                    document.head.appendChild(script);
                }} else {{
                    calculateRoute();
                }}
            }};
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=680, scrolling=False)
