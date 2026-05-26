
import pandas as pd, re
df = pd.read_parquet('d:/paper/drzero/results/iter1_baseline.parquet')

def extract_content(raw):
    parts = raw.split('<|endoftext|>')
    non_empty = [p for p in parts if p.strip()]
    return non_empty[0] if non_empty else ''

def split_conversation(rc):
    # Split into turns by im_start/im_end
    turns = re.split(r'<\|im_start\|>', rc)
    result = {}
    for t in turns:
        if t.startswith('user'):
            result['user'] = t[4:].strip()
        elif t.startswith('assistant'):
            result['assistant'] = t[9:].strip()
        elif t.startswith('system'):
            result['system'] = t[6:].strip()
    return result

# Find a 3-hop example (data_source search_zero_3)
for i, row in df.iterrows():
    if row['data_source'] == 'search_zero_3':
        rc = extract_content(row['metadata']['raw_context'])
        turns = split_conversation(rc)
        asst = turns.get('assistant', '')
        
        # Count actual tool_calls/responses in assistant output
        n_calls = asst.count('<tool_call>')
        n_responses = asst.count('<tool_response>')
        
        if n_calls >= 2 and n_responses >= 2:
            print(f'Row {i}: data_source={row["data_source"]}, calls={n_calls}, responses={n_responses}')
            print('ground_truth:', row['reward_model']['ground_truth']['target'])
            print()
            print('=== ASSISTANT OUTPUT ===')
            print(asst[:3000])
            print('...')
            print(asst[-500:])
            break

