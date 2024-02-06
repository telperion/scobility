import requests
import logging
import os
import json
from datetime import datetime as dt

def timestamp():
    return dt.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]

def setup_scrape() -> str:
    path_dst = os.path.join('itl2023_data', dt.utcnow().strftime('%Y%m%d'))
    os.makedirs(path_dst, exist_ok=True)

    # Set up logging
    logging.getLogger().handlers.clear()
    log_stamp = timestamp()
    log_path = os.path.join(path_dst, f'scobility-scrape-{log_stamp}.log')
    log_fmt = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.basicConfig(
        filename=log_path,
        encoding='utf-8',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    for handler in logging.getLogger().handlers:
        handler.setFormatter(log_fmt)

    return path_dst

def scrape_charts(path_dst: str):
    # Chart enumeration query
    charts = {}
    strikes = []
    for i in range(10000):
        try:
            r = requests.get(f'https://itl2023.groovestats.com/api/chart/{i}')
            j = r.json()
        except Exception as e:
            j = {'success': False, 'message': str(e)}

        if not j.get('success', False):
            logging.warning(f"{i:4d}: {j.get('message', '')}")
            strikes.append(i)
            if len(strikes) > 5:
                break
        else:
            strikes = []
            charts[i] = j.get('data', {})
            full_name = f"{charts[i].get('artist')} - \"{charts[i].get('title')}\""
            logging.info(f'{i:4d}: {full_name}')

    with open(os.path.join(path_dst, 'charts.json'), 'w', encoding='utf-8') as fp:
        json.dump(charts, fp)

def scrape_entrants(path_dst: str):
    p_entrants = os.path.join(path_dst, 'entrant_info')
    if not os.path.exists(p_entrants):
        os.makedirs(p_entrants)

    # Entrant enumeration query
    entrants = {}
    strikes = []
    for i in range(10000):
        try:
            r = requests.get(f'https://itl2023.groovestats.com/api/entrant/{i}')
            j = r.json()
        except Exception as e:
            j = {'success': False, 'message': str(e)}

        if not j.get('success', False):
            logging.warning(f"{i:4d}: {j.get('message', '')}")
            strikes.append(i)
            if len(strikes) > 5:
                break
        else:
            strikes = []
            entrants[i] = j.get('data', {})
            full_name = f"{entrants[i]['entrant']['name']} (ITL #{entrants[i]['entrant']['id']}, GS #{entrants[i]['entrant']['membersId']})"
            logging.info(f'{i:4d}: {full_name}')

            with open(os.path.join(p_entrants, f'{i}.json'), 'w', encoding='utf-8') as fp:
                json.dump(entrants[i], fp)

def scrape_scores(path_dst: str):
    # Scores query (examine entrants' played songs pages)
    p_scores = os.path.join(path_dst, 'song_scores')
    if not os.path.exists(p_scores):
        os.makedirs(p_scores)

    with open(os.path.join(path_dst, 'charts.json'), 'r', encoding='utf-8') as fp:
        charts = json.load(fp)

    # Entrant's played songs pages query
    scores = {}
    strikes = []
    total = 0
    for c in charts.values():
        total += 1
        if total > 10000:
            break

        i = c.get('id', 0)

        for retries in range(5):
            try:
                r = requests.post(
                    f'https://itl2023.groovestats.com/api/score/chartTopScores',
                    data={'chartHash': c['hash']}
                )
                if r.status_code > 400:
                    continue
                j = r.json()
                break
            except Exception as e:
                r = None
                j = {'success': False, 'message': str(e)}
        if r is None or r.status_code > 400:
            logging.error('Couldn\'t retrieve scores for #{i}\n{r}')

        if not j.get('success', False):
            logging.warning(f"{i:4d} (hash {c['hash']}): {j.get('message', '')}")
            strikes.append(i)
            if len(strikes) > 5:
                break
        else:
            strikes = []
            full_name = f"{c.get('artist')} - \"{c.get('title')}\""
            scores[i] = j.get('data', {}).get('leaderboard', {})
            for s in scores[i]:
                s['chartId'] = i
            logging.info(f"{i:4d} (hash {c['hash']}): {full_name}, {len(scores[i])} scores")

            with open(os.path.join(p_scores, f'{i}.json'), 'w', encoding='utf-8') as fp:
                json.dump({'scores': scores[i]}, fp)

    # Reorganize scores into individual JSON files per chart
    p_charts = os.path.join(path_dst, 'song_info')
    if not os.path.exists(p_charts):
        os.makedirs(p_charts)

    with open(os.path.join(path_dst, 'charts.json'), 'r', encoding='utf-8') as fp:
        charts = json.load(fp)

    for c in charts.values():
        i = c.get('id', 0)
        
        full_name = f"{c.get('artist')} - \"{c.get('title')}\""
        logging.info(f"{i:4d} (hash {c['hash']}): {full_name}")

        with open(os.path.join(p_charts, f'{i}.json'), 'w', encoding='utf-8') as fp:
            json.dump({'song': c}, fp)


if __name__ == '__main__':
    only_scores = False

    path_dst = setup_scrape()
    if not only_scores:
        scrape_charts(path_dst)
        scrape_entrants(path_dst)
    scrape_scores(path_dst)
 
    logging.info('Done!')