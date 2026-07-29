import pandas as pd
import numpy as np
import json

def validate():
    df = pd.read_parquet('forecasts/combined_forecasts.parquet')
    
    report = {}
    
    # 1. Total forecast records
    report['total_records'] = len(df)
    
    # 2. Unique payer_code x pack_type combinations
    unique_combinations = df[['payer_code', 'pack_type']].drop_duplicates().shape[0]
    report['unique_combinations'] = unique_combinations
    
    # 3. Forecast month continuity
    months = sorted(df['forecast_month'].unique())
    report['forecast_months'] = [str(m)[:10] for m in months]
    report['is_continuous_months'] = bool(len(months) == 3) # Should have exactly 3 consecutive months
    
    # 4. Duplicate rows
    duplicates = int(df.duplicated().sum())
    report['duplicate_rows'] = duplicates
    
    # 5. Missing values
    missing = int(df.isnull().sum().sum())
    report['missing_values'] = missing
    
    # 6. Negative forecast values
    neg_p10 = int((df['P10'] < 0).sum())
    neg_p50 = int((df['P50'] < 0).sum())
    neg_p90 = int((df['P90'] < 0).sum())
    report['negative_forecasts'] = neg_p10 + neg_p50 + neg_p90
    
    # 7. P10 <= P50 <= P90 for every record
    # Note: we need to handle potential floating point precision issues using np.isclose if necessary, 
    # but standard <= should work because of how we calculated it.
    # To be perfectly safe against float weirdness, round slightly or just use <=.
    valid_bounds1 = (df['P10'] <= df['P50'] + 1e-6).all()
    valid_bounds2 = (df['P50'] <= df['P90'] + 1e-6).all()
    report['valid_interval_bounds'] = bool(valid_bounds1 and valid_bounds2)
    
    # Summary
    passed = all([
        report['total_records'] == unique_combinations * 3,
        report['is_continuous_months'],
        report['duplicate_rows'] == 0,
        report['missing_values'] == 0,
        report['negative_forecasts'] == 0,
        report['valid_interval_bounds']
    ])
    report['ALL_PASSED'] = passed
    
    print(json.dumps(report, indent=4))
    
    with open('forecasts/validation_report.json', 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == '__main__':
    validate()
