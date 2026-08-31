

import os
import random
import math

def get_crop_agronomic_profiles():
    """
    Returns authentic agronomic parameter distributions (mean, std, min, max)
    for 22 standard agricultural crops based on agronomic benchmarks.
    """
    profiles = {
        'rice': {
            'N': (80, 10, 60, 100), 'P': (47, 8, 35, 60), 'K': (40, 5, 35, 45),
            'temp': (23.6, 2.5, 20.0, 27.5), 'hum': (82.2, 4.0, 75.0, 90.0),
            'ph': (6.4, 0.4, 5.0, 7.5), 'rain': (236.0, 30.0, 180.0, 300.0),
            'base_yield': 4.2, 'yield_std': 0.6
        },
        'maize': {
            'N': (78, 12, 60, 100), 'P': (48, 8, 35, 60), 'K': (20, 3, 15, 25),
            'temp': (22.4, 3.0, 18.0, 28.0), 'hum': (65.1, 7.0, 55.0, 80.0),
            'ph': (6.2, 0.5, 5.5, 7.5), 'rain': (85.0, 15.0, 60.0, 115.0),
            'base_yield': 5.8, 'yield_std': 0.8
        },
        'chickpea': {
            'N': (40, 10, 20, 60), 'P': (68, 7, 55, 80), 'K': (80, 5, 75, 85),
            'temp': (18.8, 1.2, 17.0, 21.0), 'hum': (16.8, 2.0, 14.0, 20.0),
            'ph': (7.3, 0.5, 6.0, 8.5), 'rain': (80.0, 10.0, 65.0, 95.0),
            'base_yield': 1.8, 'yield_std': 0.3
        },
        'kidneybeans': {
            'N': (21, 6, 15, 40), 'P': (67, 7, 55, 80), 'K': (20, 3, 15, 25),
            'temp': (20.1, 2.5, 15.0, 25.0), 'hum': (21.6, 2.5, 18.0, 26.0),
            'ph': (5.7, 0.2, 5.5, 6.0), 'rain': (105.0, 25.0, 60.0, 150.0),
            'base_yield': 1.6, 'yield_std': 0.25
        },
        'pigeonpeas': {
            'N': (21, 6, 15, 40), 'P': (68, 7, 55, 80), 'K': (20, 3, 18, 25),
            'temp': (27.7, 3.5, 24.0, 37.0), 'hum': (48.0, 7.0, 30.0, 70.0),
            'ph': (5.8, 0.7, 4.5, 7.5), 'rain': (149.0, 30.0, 90.0, 200.0),
            'base_yield': 1.5, 'yield_std': 0.25
        },
        'mothbeans': {
            'N': (21, 6, 15, 40), 'P': (48, 7, 35, 60), 'K': (20, 3, 15, 25),
            'temp': (28.2, 3.0, 24.0, 32.5), 'hum': (53.2, 7.5, 40.0, 68.0),
            'ph': (6.8, 1.2, 3.5, 9.5), 'rain': (51.2, 12.0, 30.0, 75.0),
            'base_yield': 1.2, 'yield_std': 0.2
        },
        'mungbean': {
            'N': (21, 6, 15, 40), 'P': (47, 7, 35, 60), 'K': (20, 3, 15, 25),
            'temp': (28.5, 1.2, 27.0, 30.0), 'hum': (85.5, 3.0, 80.0, 90.0),
            'ph': (6.7, 0.4, 6.2, 7.5), 'rain': (48.4, 8.0, 35.0, 60.0),
            'base_yield': 1.3, 'yield_std': 0.2
        },
        'blackgram': {
            'N': (40, 7, 35, 60), 'P': (67, 7, 55, 80), 'K': (19, 3, 15, 25),
            'temp': (29.9, 3.0, 25.0, 35.0), 'hum': (65.0, 3.5, 60.0, 70.0),
            'ph': (7.1, 0.4, 6.5, 7.8), 'rain': (67.8, 5.0, 60.0, 75.0),
            'base_yield': 1.4, 'yield_std': 0.22
        },
        'lentil': {
            'N': (19, 5, 15, 40), 'P': (68, 7, 55, 80), 'K': (19, 3, 15, 25),
            'temp': (24.5, 3.5, 18.0, 30.0), 'hum': (64.8, 3.5, 60.0, 70.0),
            'ph': (6.9, 0.6, 6.0, 8.0), 'rain': (45.7, 6.0, 35.0, 55.0),
            'base_yield': 1.5, 'yield_std': 0.25
        },
        'pomegranate': {
            'N': (19, 5, 15, 40), 'P': (19, 5, 10, 30), 'K': (40, 3, 35, 45),
            'temp': (21.8, 2.0, 18.0, 25.0), 'hum': (90.1, 2.8, 85.0, 95.0),
            'ph': (6.4, 0.5, 5.5, 7.2), 'rain': (107.5, 5.0, 100.0, 115.0),
            'base_yield': 13.5, 'yield_std': 1.8
        },
        'banana': {
            'N': (100, 12, 80, 120), 'P': (82, 8, 70, 95), 'K': (50, 3, 45, 55),
            'temp': (27.4, 1.5, 25.0, 30.0), 'hum': (80.4, 3.0, 75.0, 85.0),
            'ph': (5.98, 0.3, 5.5, 6.5), 'rain': (104.6, 9.0, 90.0, 120.0),
            'base_yield': 42.0, 'yield_std': 5.5
        },
        'mango': {
            'N': (20, 6, 15, 40), 'P': (27, 6, 15, 35), 'K': (30, 3, 25, 35),
            'temp': (31.2, 2.8, 27.0, 36.0), 'hum': (50.2, 3.0, 45.0, 55.0),
            'ph': (5.77, 0.7, 4.5, 7.0), 'rain': (94.7, 6.0, 85.0, 105.0),
            'base_yield': 12.0, 'yield_std': 1.9
        },
        'grapes': {
            'N': (23, 7, 15, 40), 'P': (132, 8, 120, 145), 'K': (200, 4, 195, 205),
            'temp': (23.8, 8.0, 10.0, 42.0), 'hum': (81.9, 1.8, 80.0, 85.0),
            'ph': (6.0, 0.3, 5.5, 6.5), 'rain': (69.6, 3.5, 65.0, 75.0),
            'base_yield': 21.5, 'yield_std': 2.8
        },
        'watermelon': {
            'N': (99, 12, 80, 120), 'P': (17, 6, 5, 30), 'K': (50, 3, 45, 55),
            'temp': (25.6, 1.0, 24.0, 27.0), 'hum': (85.2, 3.0, 80.0, 90.0),
            'ph': (6.5, 0.3, 6.0, 7.0), 'rain': (50.8, 6.0, 40.0, 60.0),
            'base_yield': 36.0, 'yield_std': 4.5
        },
        'muskmelon': {
            'N': (100, 12, 80, 120), 'P': (17, 6, 5, 30), 'K': (50, 3, 45, 55),
            'temp': (28.6, 1.0, 27.0, 30.0), 'hum': (92.3, 1.5, 90.0, 95.0),
            'ph': (6.36, 0.25, 6.0, 6.8), 'rain': (24.7, 3.0, 20.0, 30.0),
            'base_yield': 26.0, 'yield_std': 3.2
        },
        'apple': {
            'N': (21, 6, 15, 40), 'P': (134, 7, 120, 145), 'K': (200, 4, 195, 205),
            'temp': (22.6, 1.0, 21.0, 24.0), 'hum': (92.3, 1.5, 90.0, 95.0),
            'ph': (5.93, 0.3, 5.5, 6.5), 'rain': (112.7, 7.0, 100.0, 125.0),
            'base_yield': 18.5, 'yield_std': 2.4
        },
        'orange': {
            'N': (20, 6, 15, 40), 'P': (16, 6, 5, 30), 'K': (10, 3, 5, 15),
            'temp': (22.8, 7.0, 10.0, 35.0), 'hum': (92.2, 1.6, 90.0, 95.0),
            'ph': (7.01, 0.6, 6.0, 8.0), 'rain': (110.5, 6.0, 100.0, 120.0),
            'base_yield': 21.0, 'yield_std': 2.7
        },
        'papaya': {
            'N': (50, 12, 35, 75), 'P': (59, 8, 45, 70), 'K': (50, 3, 45, 55),
            'temp': (33.7, 6.0, 23.0, 44.0), 'hum': (92.4, 1.6, 90.0, 95.0),
            'ph': (6.74, 0.15, 6.5, 7.0), 'rain': (142.6, 40.0, 90.0, 250.0),
            'base_yield': 48.0, 'yield_std': 6.0
        },
        'coconut': {
            'N': (22, 6, 15, 40), 'P': (17, 6, 5, 30), 'K': (30, 3, 25, 35),
            'temp': (27.4, 1.2, 25.0, 29.0), 'hum': (95.0, 2.5, 90.0, 100.0),
            'ph': (5.98, 0.3, 5.5, 6.5), 'rain': (175.7, 30.0, 130.0, 230.0),
            'base_yield': 11.5, 'yield_std': 1.6
        },
        'cotton': {
            'N': (118, 12, 100, 140), 'P': (46, 8, 35, 60), 'K': (20, 3, 15, 25),
            'temp': (24.0, 1.2, 22.0, 26.0), 'hum': (79.8, 7.0, 60.0, 85.0),
            'ph': (6.91, 0.6, 6.0, 8.0), 'rain': (80.4, 12.0, 60.0, 100.0),
            'base_yield': 2.4, 'yield_std': 0.35
        },
        'jute': {
            'N': (78, 12, 60, 100), 'P': (47, 8, 35, 60), 'K': (40, 3, 35, 45),
            'temp': (25.0, 1.0, 23.0, 26.0), 'hum': (79.6, 6.0, 70.0, 90.0),
            'ph': (6.73, 0.4, 6.0, 7.5), 'rain': (174.8, 15.0, 150.0, 200.0),
            'base_yield': 3.1, 'yield_std': 0.4
        },
        'coffee': {
            'N': (101, 12, 80, 120), 'P': (29, 7, 15, 40), 'K': (30, 3, 25, 35),
            'temp': (25.5, 1.5, 23.0, 28.0), 'hum': (58.9, 6.0, 50.0, 70.0),
            'ph': (6.79, 0.45, 6.0, 7.5), 'rain': (158.1, 25.0, 115.0, 200.0),
            'base_yield': 1.9, 'yield_std': 0.3
        }
    }
    return profiles

def _clip(val, min_v, max_v):
    return max(min_v, min(max_v, val))

def generate_crop_recommendation_dataset(output_path='data/Crop_recommendation.csv', samples_per_crop=100, seed=42):
    """
    Generates 2,200 records (100 per crop * 22 crops) for multi-class crop classification.
    """
    random.seed(seed)
    profiles = get_crop_agronomic_profiles()
    rows = ["N,P,K,temperature,humidity,ph,rainfall,label"]
    record_list = []

    for crop, prof in profiles.items():
        for _ in range(samples_per_crop):
            n = int(round(_clip(random.gauss(prof['N'][0], prof['N'][1]), prof['N'][2], prof['N'][3])))
            p = int(round(_clip(random.gauss(prof['P'][0], prof['P'][1]), prof['P'][2], prof['P'][3])))
            k = int(round(_clip(random.gauss(prof['K'][0], prof['K'][1]), prof['K'][2], prof['K'][3])))
            
            temp = round(_clip(random.gauss(prof['temp'][0], prof['temp'][1]), prof['temp'][2], prof['temp'][3]), 5)
            hum = round(_clip(random.gauss(prof['hum'][0], prof['hum'][1]), prof['hum'][2], prof['hum'][3]), 5)
            ph = round(_clip(random.gauss(prof['ph'][0], prof['ph'][1]), prof['ph'][2], prof['ph'][3]), 5)
            rain = round(_clip(random.gauss(prof['rain'][0], prof['rain'][1]), prof['rain'][2], prof['rain'][3]), 5)
            
            record_list.append(f"{n},{p},{k},{temp},{hum},{ph},{rain},{crop}")

    random.shuffle(record_list)
    rows.extend(record_list)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(rows) + "\n")
    print(f"Generated {len(record_list)} records for Crop Recommendation dataset at '{output_path}'.")

def generate_crop_yield_dataset(output_path='data/crop_yield.csv', samples_per_crop=120, seed=101):
    """
    Generates 2,640 records (120 per crop * 22 crops) for supervised yield regression modeling.
    """
    random.seed(seed)
    profiles = get_crop_agronomic_profiles()
    rows = ["Crop,N,P,K,temperature,humidity,ph,rainfall,Area_ha,Yield_tonnes_per_ha,Total_Production_tonnes"]
    record_list = []

    for crop, prof in profiles.items():
        base_yield = prof['base_yield']
        yield_std = prof['yield_std']

        for _ in range(samples_per_crop):
            area_ha = round(random.uniform(0.5, 12.0), 2)

            n = int(round(_clip(random.gauss(prof['N'][0], prof['N'][1] * 1.3), 5, 150)))
            p = int(round(_clip(random.gauss(prof['P'][0], prof['P'][1] * 1.3), 5, 155)))
            k = int(round(_clip(random.gauss(prof['K'][0], prof['K'][1] * 1.3), 5, 215)))
            
            temp = round(_clip(random.gauss(prof['temp'][0], prof['temp'][1] * 1.2), 8.0, 46.0), 2)
            hum = round(_clip(random.gauss(prof['hum'][0], prof['hum'][1] * 1.2), 12.0, 99.0), 2)
            ph = round(_clip(random.gauss(prof['ph'][0], prof['ph'][1] * 1.2), 3.5, 9.5), 2)
            rain = round(_clip(random.gauss(prof['rain'][0], prof['rain'][1] * 1.2), 15.0, 350.0), 2)

            dev_n = abs(n - prof['N'][0]) / (prof['N'][1] * 2.5 + 1e-5)
            dev_p = abs(p - prof['P'][0]) / (prof['P'][1] * 2.5 + 1e-5)
            dev_k = abs(k - prof['K'][0]) / (prof['K'][1] * 2.5 + 1e-5)
            dev_temp = abs(temp - prof['temp'][0]) / (prof['temp'][1] * 2.5 + 1e-5)
            dev_hum = abs(hum - prof['hum'][0]) / (prof['hum'][1] * 2.5 + 1e-5)
            dev_ph = abs(ph - prof['ph'][0]) / (prof['ph'][1] * 2.5 + 1e-5)
            dev_rain = abs(rain - prof['rain'][0]) / (prof['rain'][1] * 2.5 + 1e-5)

            penalty = 0.05 * dev_n + 0.05 * dev_p + 0.05 * dev_k + 0.06 * dev_temp + 0.04 * dev_hum + 0.06 * dev_ph + 0.07 * dev_rain
            efficiency = max(0.4, 1.15 - penalty)

            noise = random.gauss(0, yield_std * 0.12)
            yield_tonnes_ha = max(0.2, (base_yield * efficiency) + noise)
            yield_tonnes_ha = round(yield_tonnes_ha, 3)

            total_production = round(yield_tonnes_ha * area_ha, 3)

            record_list.append(f"{crop},{n},{p},{k},{temp},{hum},{ph},{rain},{area_ha},{yield_tonnes_ha},{total_production}")

    random.shuffle(record_list)
    rows.extend(record_list)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(rows) + "\n")
    print(f"Generated {len(record_list)} records for Crop Yield dataset at '{output_path}'.")

if __name__ == '__main__':
    print("Generating agricultural benchmark datasets...")
    generate_crop_recommendation_dataset()
    generate_crop_yield_dataset()
    print("\nDataset generation completed successfully!")
