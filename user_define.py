

import os
import sys
import json
import pandas as pd
from predict import CropAndYieldPredictor

def print_banner():
    print("\n" + "=" * 75)
    print("🌾 AGRO-AI: USER-DEFINED PREDICTION & CUSTOM SCENARIO ENGINE")
    print("=" * 75)

def get_float(prompt, default_val, min_val=0.0, max_val=10000.0):
    while True:
        try:
            val_str = input(f" 👉 {prompt} [Default: {default_val}]: ").strip()
            if not val_str:
                return float(default_val)
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            print(f"    ⚠️ Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("    ⚠️ Invalid number. Please enter a valid numerical value.")

def custom_single_prediction(predictor):
    print("\n" + "-" * 75)
    print("📝 STEP 1: Enter Your Custom Soil & Environmental Measurements")
    print("-" * 75)

    n = get_float("Soil Available Nitrogen (N) in kg/ha", 85.0, 0.0, 300.0)
    p = get_float("Soil Available Phosphorus (P) in kg/ha", 48.0, 0.0, 300.0)
    k = get_float("Soil Available Potassium (K) in kg/ha", 40.0, 0.0, 300.0)
    temp = get_float("Mean Ambient Temperature in °C", 24.0, 0.0, 60.0)
    hum = get_float("Relative Humidity in %", 84.0, 5.0, 100.0)
    ph = get_float("Soil pH Level (0 - 14 scale)", 6.4, 3.0, 10.0)
    rain = get_float("Seasonal / Annual Rainfall in mm", 240.0, 0.0, 2000.0)
    area = get_float("Farm Land Area in Hectares", 3.0, 0.1, 10000.0)

    print("\n🔄 Running Machine Learning Multi-Stage Prediction...")
    res = predictor.predict_all(n=n, p=p, k=k, temp=temp, hum=hum, ph=ph, rain=rain, area_ha=area)
    rec = res['stage1_recommendation']
    yld = res['stage2_yield_forecast']

    print("\n" + "=" * 75)
    print("📊 USER-DEFINED PREDICTION REPORT")
    print("=" * 75)
    print(f"🌱 STAGE 1: RECOMMENDED OPTIMAL CROP")
    print(f"   • Recommended Variety: [{rec['recommended_crop'].upper()}]")
    print(f"   • Model Confidence:    {rec['confidence_percent']:.2f}%")
    print(f"   • Top Alternatives:    " + ", ".join([f"{c['crop'].title()} ({c['confidence']}%)" for c in rec['top_candidates'][1:]]))

    print(f"\n📈 STAGE 2: YIELD & HARVEST FORECAST")
    print(f"   • Yield Efficiency:    {yld['yield_tonnes_per_ha']} Tonnes / Hectare")
    print(f"   • Total Farm Harvest:  {yld['total_production_tonnes']} Metric Tonnes (on {area} ha)")
    print(f"   • In Quintals:         {yld['total_production_quintals']} Quintals")
    print(f"   • In Kilograms:        {yld['total_production_kg']:,.0f} kg")

    # Agronomic NPK check
    npk_tot = n + p + k + 1e-6
    print(f"\n🧪 SOIL NUTRIENT ANALYSIS:")
    print(f"   • NPK Ratio: N({(n/npk_tot)*100:.1f}%) : P({(p/npk_tot)*100:.1f}%) : K({(k/npk_tot)*100:.1f}%)")
    if ph < 5.5:
        print("   • Soil Condition: Acidic soil. Consider agricultural lime application.")
    elif ph > 7.8:
        print("   • Soil Condition: Alkaline soil. Consider gypsum / organic sulfur.")
    else:
        print("   • Soil Condition: Ideal neutral pH range for nutrient bio-availability.")
    print("=" * 75)

    save_choice = input("\n💾 Save this user report as JSON? (y/n) [Default: n]: ").strip().lower()
    if save_choice == 'y':
        filename = "user_prediction_output.json"
        with open(filename, 'w') as f:
            json.dump(res, f, indent=4)
        print(f"✅ Report saved to '{filename}'.")

def custom_batch_prediction(predictor):
    print("\n" + "-" * 75)
    print("📂 BATCH USER-DEFINED PREDICTION")
    print("-" * 75)
    csv_file = input("Enter path to your custom CSV file (or press ENTER to use sample batch): ").strip()

    if not csv_file:
        sample_rows = [
            {'N': 85, 'P': 48, 'K': 40, 'temperature': 24.0, 'humidity': 84.0, 'ph': 6.4, 'rainfall': 240.0, 'Area_ha': 3.0},
            {'N': 20, 'P': 130, 'K': 200, 'temperature': 22.0, 'humidity': 92.0, 'ph': 6.0, 'rainfall': 110.0, 'Area_ha': 2.5},
            {'N': 40, 'P': 65, 'K': 80, 'temperature': 19.0, 'humidity': 16.0, 'ph': 7.2, 'rainfall': 80.0, 'Area_ha': 5.0},
            {'N': 120, 'P': 45, 'K': 20, 'temperature': 24.5, 'humidity': 78.0, 'ph': 6.8, 'rainfall': 80.0, 'Area_ha': 4.0}
        ]
        df = pd.DataFrame(sample_rows)
        print("Loaded sample batch of 4 farms.")
    else:
        if not os.path.exists(csv_file):
            print(f"❌ File not found at '{csv_file}'.")
            return
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} farm records from '{csv_file}'.")

    results_df = predictor.predict_batch(df)
    print("\n--- Batch Prediction Results ---")
    print(results_df[['N', 'P', 'K', 'Recommended_Crop', 'Confidence_Percent', 'Yield_Tonnes_Per_Ha', 'Total_Production_Tonnes']])

    out_csv = "user_batch_prediction_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ All batch predictions exported to '{out_csv}'.")

def custom_add_crop(predictor):
    print("\n" + "-" * 75)
    print("➕ DEFINE A NEW CUSTOM CROP VARIETY")
    print("-" * 75)
    name = input("Enter new crop name (e.g. Dragon Fruit, Avocado): ").strip()
    if not name:
        name = "Dragon Fruit"

    n = get_float("Optimal Nitrogen (N) kg/ha", 70.0)
    p = get_float("Optimal Phosphorus (P) kg/ha", 35.0)
    k = get_float("Optimal Potassium (K) kg/ha", 45.0)
    temp = get_float("Optimal Temperature (°C)", 28.0)
    hum = get_float("Optimal Humidity (%)", 70.0)
    ph = get_float("Optimal Soil pH", 6.5)
    rain = get_float("Optimal Rainfall (mm)", 90.0)
    base_yield = get_float("Base Expected Yield (Tonnes/Ha)", 12.5)

    predictor.add_user_defined_crop(name, n, p, k, temp, hum, ph, rain, base_yield)
    print(f"\n✅ Custom Crop '{name.title()}' successfully registered in the ML engine!")

def main_menu():
    predictor = CropAndYieldPredictor()
    while True:
        print_banner()
        print("Select an option:")
        print(" [1] ✍️  Custom Single Farm Prediction (Interactive Prompt)")
        print(" [2] 📂  Batch User-Defined CSV Processing")
        print(" [3] ➕  Register a Custom / User-Defined Crop")
        print(" [4] 🧪  Run Standard Benchmark Test Scenarios")
        print(" [5] 🚪  Exit")
        print("=" * 75)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            custom_single_prediction(predictor)
        elif choice == '2':
            custom_batch_prediction(predictor)
        elif choice == '3':
            custom_add_crop(predictor)
        elif choice == '4':
            from predict import run_test_scenarios
            run_test_scenarios()
        elif choice == '5' or choice.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Exiting AgroAI Engine. Happy Farming!\n")
            break
        else:
            print("⚠️ Invalid choice. Please select 1, 2, 3, 4, or 5.")

        input("\nPress ENTER to return to menu...")

if __name__ == '__main__':
    main_menu()
