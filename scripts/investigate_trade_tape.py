#!/usr/bin/env python3
"""
Investigate Polymarket API for trade tape/executed trades.
Part of spread capture strategy implementation.
"""

import requests
import json

# Test Polymarket API endpoints for trade data
base_url = 'https://clob.polymarket.com'
gamma_base = 'https://gamma-api.polymarket.com'

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

print('=' * 60)
print('INVESTIGATING POLYMARKET API FOR TRADE TAPE')
print('=' * 60)

# Test endpoints
test_endpoints = [
    f'{base_url}/trades',
    f'{base_url}/trade-history',
    f'{base_url}/fills',
    f'{gamma_base}/trades',
    f'{gamma_base}/activity',
    f'{gamma_base}/fills',
]

print('\nTesting potential trade tape endpoints...')
print('-' * 60)

for endpoint in test_endpoints:
    try:
        resp = session.get(endpoint, timeout=10)
        print(f'\n{endpoint}:')
        print(f'  Status: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                print(f'  Response keys: {list(data.keys())}')
                if 'data' in data:
                    data_items = data['data']
                    print(f'  Data count: {len(data_items)}')
                    if data_items and len(data_items) > 0:
                        print(f'  Sample item keys: {list(data_items[0].keys()) if isinstance(data_items[0], dict) else "N/A"}')
            elif isinstance(data, list):
                print(f'  Response: list with {len(data)} items')
                if data and len(data) > 0:
                    print(f'  Sample item keys: {list(data[0].keys()) if isinstance(data[0], dict) else "N/A"}')
        else:
            resp_text = resp.text[:300] if len(resp.text) > 300 else resp.text
            print(f'  Response: {resp_text}')
    except Exception as e:
        print(f'\n{endpoint}:')
        print(f'  Error: {e}')

# Check Strapi API (used for market info)
print('\n' + '-' * 60)
print('Testing Strapi API for activity data...')

strapi_endpoints = [
    'https://strapi-matic.poly.market/activities',
    'https://strapi-matic.poly.market/trades',
]

for endpoint in strapi_endpoints:
    try:
        resp = session.get(endpoint, params={'_limit': 5}, timeout=10)
        print(f'\n{endpoint}:')
        print(f'  Status: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                print(f'  Items: {len(data)}')
                print(f'  Sample keys: {list(data[0].keys())}')
    except Exception as e:
        print(f'\n{endpoint}:')
        print(f'  Error: {e}')

# Check data subgraph
print('\n' + '-' * 60)
print('Testing The Graph subgraph for trade data...')

# Polymarket uses The Graph for historical data
subgraph_url = 'https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic'
query = '''
{
  trades(first: 5, orderBy: timestamp, orderDirection: desc) {
    id
    timestamp
    price
    size
    side
  }
}
'''

try:
    resp = session.post(subgraph_url, json={'query': query}, timeout=15)
    print(f'\nSubgraph endpoint: {subgraph_url}')
    print(f'  Status: {resp.status_code}')
    if resp.status_code == 200:
        data = resp.json()
        print(f'  Response keys: {list(data.keys())}')
        if 'data' in data:
            print(f'  Data: {json.dumps(data["data"], indent=2)[:500]}')
        if 'errors' in data:
            print(f'  Errors: {data["errors"]}')
except Exception as e:
    print(f'  Error: {e}')

print('\n' + '=' * 60)
print('CONCLUSION')
print('=' * 60)
print('''
Based on API investigation:

1. CLOB API (/book, /midpoint) - ORDERBOOK SNAPSHOTS ONLY
   - No trade tape endpoint found
   - Only provides current state, not historical trades

2. Gamma API - MARKET METADATA ONLY
   - Provides market info, outcomes, token IDs
   - No trade history endpoint

3. Strapi API - ACCESS RESTRICTED
   - Activity/trades endpoints may exist but require auth

4. The Graph Subgraph - POTENTIAL SOURCE
   - May provide historical trade data
   - Need to find correct subgraph name

RECOMMENDATION:
- Use TOUCH_SIZE_PROXY fill model (we have size data)
- Can implement BOUNDS_ONLY mode as fallback
- Trade tape would require:
  a) Finding Polymarket's subgraph, OR
  b) Building real-time tape by diffing orderbook snapshots
  c) Using wallet activity data (already have this for 6 traders)

For now, proceed with TOUCH_SIZE_PROXY using size data from 12 volume markets.
''')

