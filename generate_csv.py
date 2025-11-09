# ================================================================
# GENERATE CSV FROM CLINICAL READINGS
# Run this BEFORE the main lung cancer detection script
# ================================================================

import pandas as pd
import os

CLINICAL_PATH = 'data/ClinicalReadings'
OUTPUT_CSV = 'data/output.csv'

def extract_clinical_data(clinical_path):
    """Extract patient data from clinical reading text files"""
    data = []
    
    if not os.path.exists(clinical_path):
        print(f"❌ Clinical readings folder not found: {clinical_path}")
        return None
    
    txt_files = [f for f in os.listdir(clinical_path) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} clinical reading files")
    
    for filename in txt_files:
        filepath = os.path.join(clinical_path, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read().lower()
                
                # Extract disease/diagnosis information
                disease = 'unknown'
                
                # Check for specific disease keywords
                if 'normal' in content:
                    disease = 'normal'
                elif any(word in content for word in ['cancer', 'malignant', 'tumor', 'carcinoma', 'malignancy']):
                    disease = 'cancer'
                elif any(word in content for word in ['tuberculosis', 'tb']):
                    disease = 'tuberculosis'
                elif any(word in content for word in ['pneumonia', 'infection']):
                    disease = 'pneumonia'
                elif any(word in content for word in ['abnormal', 'suspicious', 'nodule', 'lesion', 'opacity']):
                    disease = 'abnormal'
                
                data.append({
                    'File Name': filename,
                    'Disease': disease
                })
                
        except Exception as e:
            print(f"⚠ Error reading {filename}: {e}")
    
    df = pd.DataFrame(data)
    return df

# Main execution
print("="*60)
print("EXTRACTING CLINICAL DATA")
print("="*60)

clinical_df = extract_clinical_data(CLINICAL_PATH)

if clinical_df is not None and len(clinical_df) > 0:
    # Save to CSV
    clinical_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ CSV saved to: {OUTPUT_CSV}")
    print(f"✓ Total records: {len(clinical_df)}")
    
    print(f"\n{'='*60}")
    print("DISEASE DISTRIBUTION")
    print("="*60)
    disease_counts = clinical_df['Disease'].value_counts()
    for disease, count in disease_counts.items():
        percentage = (count / len(clinical_df)) * 100
        print(f"{disease.capitalize():15s}: {count:4d} ({percentage:5.2f}%)")
    
    # Show sample data
    print(f"\n{'='*60}")
    print("SAMPLE DATA (first 10 records)")
    print("="*60)
    print(clinical_df.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("✓ CSV GENERATION COMPLETE!")
    print("="*60)
    print("You can now run the main lung-cancer-detection.py script")
    
else:
    print("❌ Could not generate CSV from clinical readings")
    print("Please check if the ClinicalReadings folder exists and contains .txt files")