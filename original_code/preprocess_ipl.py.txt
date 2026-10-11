import json
import os
import pandas as pd

RAW_DATA_DIR = "./data/raw/ipl"
OUTPUT_FILE = "./data/processed/player_containers.json"

def calculate_dream11_points(stats):
    runs = stats.get('runs', 0)
    pts += runs * 1
    pts += stats.get('fours', 0) * 1
    pts += stats.get('sixes', 0) * 2
    if runs >= 100: pts += 16
    elif runs >= 50: pts += 8
    elif runs >= 30: pts += 4
    if runs == 0 and stats.get('balls_faced', 0) > 0 and stats.get('role') != 'BOWL':
        pts -= 2 
    
    wickets = stats.get('wickets', 0)
    pts += wickets * 25
    if wickets >= 5: pts += 16
    elif wickets >= 4: pts += 8
    elif wickets >= 3: pts += 4
    pts += stats.get('maidens', 0) * 12

   pts += stats.get('catches', 0) * 8
    pts += stats.get('stumpings', 0) * 12
    pts += stats.get('run_outs', 0) * 6
    
    return pts


def parse_match(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    info = data.get('info', {})
    match_date = info.get('dates', ['2024-01-01'])[0]
    venue = info.get('venue', 'Unknown')
    
    player_stats = {}
    for innings in data.get('innings', []):
        for over in innings.get('overs', []):
            for delivery in over.get('deliveries', []):
                batter = delivery['batter']
                bowler = delivery['bowler']
                runs = delivery['runs']['batter']
                
                for p in [batter, bowler]:
                    if p not in player_stats:
                        player_stats[p] = {'runs': 0, 'fours': 0, 'sixes': 0, 'wickets': 0, 'catches': 0}
                
                player_stats[batter]['runs'] += runs
                if runs == 4: player_stats[batter]['fours'] += 1
                elif runs == 6: player_stats[batter]['sixes'] += 1
                
                if 'wickets' in delivery:
                    for w in delivery['wickets']:
                        if w['kind'] not in ['run out', 'retired hurt']:
                            player_stats[bowler]['wickets'] += 1
                        for fielder in w.get('fielders', []):
                            f_name = fielder.get('name')
                            if f_name:
                                if f_name not in player_stats:
                                    player_stats[f_name] = {'runs': 0, 'fours': 0, 'sixes': 0, 'wickets': 0, 'catches': 0}
                                player_stats[f_name]['catches'] += 1

    records = []
    for player, stats in player_stats.items():
        records.append({
            'date': match_date,
            'venue': venue,
            'player_name': player,
            'runs': stats['runs'],
            'wickets': stats['wickets'],
            'catches': stats['catches'],
            'dream11_points': calculate_dream11_points(stats)
        })
    return records

def build_containers():
    all_records = []
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    print(f"Processing {len(files)} match JSONs...")

    for file in files:
        try:
            records = parse_match(os.path.join(RAW_DATA_DIR, file))
            all_records.extend(records)
        except Exception:
            continue

    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['player_name', 'date'])

    containers = []
    for player, p_df in df.groupby('player_name'):
        p_df = p_df.reset_index(drop=True)
        for i in range(len(p_df)):
            if i < 2: continue  # Need past 2 matches for features
            past = p_df.iloc[max(0, i-5):i]
            target = p_df.iloc[i]
            
            containers.append({
                "player_name": player,
                "match_date": str(target['date'].date()),
                "venue": target['venue'],
                "features": {
                    "avg_pts_last_5": float(past['dream11_points'].mean()),
                    "max_pts_last_5": float(past['dream11_points'].max()),
                    "avg_runs_last_5": float(past['runs'].mean()),
                    "avg_wickets_last_5": float(past['wickets'].mean())
                },
                "is_top_performer": 1 if target['dream11_points'] >= 40 else 0
            })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(containers, f, indent=2)
    print(f"Saved {len(containers)} containers to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_containers()