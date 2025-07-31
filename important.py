import pandas as pd
df = pd.read_csv('/Users/nikolas/Downloads/transcription_combined(8).csv')

text = ' '.join(df['Segment'])

# SAve this as talk1

with open('Talk7.txt', 'w') as f:
    f.write(text)