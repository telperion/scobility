import requests
import logging
import os
import json
from datetime import datetime as dt
from typing import Union
from time import sleep

def timestamp():
    return dt.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]

def setup_scrape() -> str:
    path_dst = os.path.join('smx_data', dt.utcnow().strftime('%Y%m%d'))
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

def is_valid_user_response(response_json) -> bool:
    if 'users' not in response_json:
        return False
    if len(response_json['users']) == 0:
        return False
    return True

def is_valid_user(response_json) -> bool:
    if not is_valid_user_response(response_json):
        return False
    if 'id' not in response_json['users'][0]:
        return False
    return True


_ARBITRARILY_HIGH_USER_ID = 69420
def scrape(path_dst: str):
    try:
        os.makedirs(os.path.join(path_dst, 'data'), exist_ok=True)
        r = requests.get('https://statmaniax.com/api/get_song_data', timeout=30)
        j = r.json()
        with open(os.path.join(path_dst, 'smx.json'), 'w') as fp:
            json.dump(j, fp)
        
        # TODO: I should talk to Cube about how to implement this properly LOL
        user_id_limit = _ARBITRARILY_HIGH_USER_ID
        r = requests.get(f'https://statmaniax.com/api/users/{_ARBITRARILY_HIGH_USER_ID}', timeout=30)
        j = r.json()
        if not is_valid_user_response(j):
            raise ValueError(f"Oh no! Couldn't query the statmaniax database for arbitrarily high user ID {_ARBITRARILY_HIGH_USER_ID}!")
        if is_valid_user(j):
            raise ValueError(f"Oh no! The playerbase has grown so large that arbitrarily high user ID {_ARBITRARILY_HIGH_USER_ID} is a real person now!")
        else:
            user_id_limit = j['users'][0]['rank']
        
        for user_id in range(user_id_limit):
            user_scores = {'scores': {}}
            r = requests.get(f'https://statmaniax.com/api/users/{user_id}', timeout=30)
            j = r.json()
            sleep(0.01)
            if not is_valid_user(j):
                logging.warning(f"User #{user_id:6d} doesn't exist")
                continue
            if float(j['users'][0]['total_score']) <= 0:
                logging.warning(f"User #{user_id:6d} doesn't have any scores. Rank: #{j['users'][0]['rank']:6d}")
                continue
                
            for chart_mode in ['beginner', 'easy', 'hard', 'wild', 'dual', 'full']:
                r = requests.get(f'https://statmaniax.com/api/get_user_highscores_info/{user_id}/{chart_mode}', timeout=30)
                j = r.json()
                sleep(0.01)
                if 'scores' in j:
                    logging.info(f"User #{user_id:6d} has {len(j['scores']):6d} scores on {chart_mode:9s} mode")
                    user_scores['scores'].update(j['scores'])
            if len(user_scores['scores']) > 0:
                with open(os.path.join(path_dst, 'data', f'{user_id:06d}.json'), 'w') as fp:
                    json.dump(j, fp)

    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    path_dst = setup_scrape()
    scrape(path_dst)
 
    logging.info('Done!')

