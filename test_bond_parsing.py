# Test bond data parsing with your real data
import pandas as pd
import io

# Your actual bond data (first few rows)
test_data = """Issuer,Sector,Country,Currency,ISIN,Bond Type,Issue Date,Maturity Date,Coupon Type,Coupon Rate (%),Frequency,Day Count Convention,Clean Price,Dirty Price,Yield to Maturity (%),Yield to Worst (%),Current Yield (%),Spread to Benchmark (bps),OAS (bps),Z-Spread (bps),Credit Rating,Duration (Modified),Convexity,DV01,Probability of Default (1Y %),Last Trade Date,Bid Price,Ask Price,Bid Yield (%),Ask Yield (%),Trading Volume (USD),1W % Price Change,1M % Price Change
Corp 1,Financials,USA,EUR,XS655050052120,Fixed,9/6/2015,9/4/2022,Fixed,5.16,Semiannual,ACT/360,95.71,96.16,2.13,1.94,5.39,172.23,159.55,180.2,BBB,7.05,2.86,0.564,0.94,45865,95.45,95.94,2.26,2.01,2859073.73,-0.32,0.58
Corp 1,Financials,USA,EUR,XS128355454532,Fixed,7/8/2019,7/5/2030,Fixed,1.23,Annual,ACT/360,95.61,97.02,7.45,7.05,1.29,135.77,157.42,139.66,BB,5.33,1.89,0.4264,0.1,45861,95.19,95.73,7.59,7.48,9592831.54,0.37,1.8"""

# Test the parsing function
from dashboard.data import _parse_bond_data

# Convert to DataFrame
df_raw = pd.read_csv(io.StringIO(test_data))

try:
    df_parsed = _parse_bond_data(df_raw)
    print("✅ Bond data parsing successful!")
    print(f"📊 Parsed {len(df_parsed)} bonds")
    print(f"🔢 Columns: {list(df_parsed.columns)}")
    
    # Show key metrics
    if 'Duration' in df_parsed.columns:
        print(f"📈 Duration range: {df_parsed['Duration'].min():.2f} - {df_parsed['Duration'].max():.2f}")
    if 'Yield' in df_parsed.columns:
        print(f"💰 Yield range: {df_parsed['Yield'].min():.3f} - {df_parsed['Yield'].max():.3f}")
        
except Exception as e:
    print(f"❌ Bond data parsing failed: {e}")
    import traceback
    traceback.print_exc()