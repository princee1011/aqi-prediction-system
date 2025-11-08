from config import AQI_CATEGORIES


# ============================================================
# 📌 CPCB Breakpoints for AQI Calculation
# ============================================================
BREAKPOINTS = {
    'pm25': [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    'pm10': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 1000, 401, 500),
    ],
    'o3': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    'no2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    'so2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 10000, 401, 500),
    ]
}


# ============================================================
# ✅ Sub-Index Calculation
# ============================================================
def compute_sub_index(pollutant, value):
    if value is None:
        return 0

    for (BP_lo, BP_hi, I_lo, I_hi) in BREAKPOINTS[pollutant]:
        if BP_lo <= value <= BP_hi:
            return round(((I_hi - I_lo) / (BP_hi - BP_lo)) * (value - BP_lo) + I_lo)

    return 0


# ============================================================
# ✅ Determine AQI Category, Color, Emoji
# ============================================================
def classify_aqi(aqi):
    for (low, high, label, color, emoji) in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label, color, emoji
    return "Unknown", "gray", "❓"


# ============================================================
# 🌍 Main AQI Calculation Function
# ============================================================
def calculate_aqi_from_pollutants(pollutants):
    sub_indices = {}

    for pollutant, value in pollutants.items():
        sub_indices[pollutant] = compute_sub_index(pollutant, value)

    aqi = max(sub_indices.values())
    dominant = max(sub_indices, key=sub_indices.get)

    category, color, emoji = classify_aqi(aqi)

    # ✅ Health Guidance
    health_info = {
        "Good": "Air quality is considered satisfactory.",
        "Satisfactory": "Minor breathing discomfort for sensitive people.",
        "Moderate": "Breathing discomfort for people with lung disease.",
        "Poor": "Respiratory issues likely for most people.",
        "Very Poor": "Serious respiratory problems for all.",
        "Severe": "Health emergency. Avoid outdoor exposure."
    }

    precautions = {
        "Good": ["Enjoy outdoor activities 🌿"],
        "Satisfactory": ["No major precautions needed 😊"],
        "Moderate": ["Wear mask outdoors 😷", "Avoid outdoor exercise"],
        "Poor": ["Limit outdoor activity", "Use air purifier indoors"],
        "Very Poor": ["Avoid going out", "Use N95 mask 🤧"],
        "Severe": ["Stay indoors 🚨", "Seek medical attention if needed"]
    }

    return {
        "aqi": aqi,
        "category": category,
        "color": color,
        "emoji": emoji,
        "dominant_pollutant": dominant.upper(),
        "sub_indices": sub_indices,
        "health_implications": health_info[category],
        "precautionary_actions": precautions[category]
    }


# ============================================================
# 🔮 Daily AQI Forecast from Predicted Pollutants
# ============================================================
def calculate_daily_aqi_from_predictions(pred_dict):
    results = []
    days = len(next(iter(pred_dict.values())))

    for day in range(days):
        pollutants_day = {p: pred_dict[p][day] for p in pred_dict}
        res = calculate_aqi_from_pollutants(pollutants_day)
        results.append({
            "day": day + 1,
            "aqi": res["aqi"],
            "category": res["category"],
            "emoji": res["emoji"],
            "dominant_pollutant": res["dominant_pollutant"]
        })

    return results
