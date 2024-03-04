import azure.functions as func

import logging
import os
from datetime import datetime as dt, timezone as tz
from typing import Union, Dict

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, TypeAdapter

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import bindparam
from sqlalchemy import MetaData
from sqlalchemy.engine import URL

import numpy as np

from db_types import Catalog, Chart, Player, Relationship, Score
import scobility_lite

# http://blog.pamelafox.org/2022/11/fastapi-on-azure-functions-with-azure.html
# if os.getenv("FUNCTIONS_WORKER_RUNTIME"):
#     api = FastAPI(
#         servers=[
#             {"url": "/api",
#             "description": "API"}
#         ],
#         root_path="/public",
#         root_path_in_servers=False
#     )
# else:
#     api = FastAPI()
api = FastAPI(
    root_path="/api"
)


## Helper functions
_TOO_MANY_THINGS = 69420

_HASH_LEN = 16
def cid(q: Union[int, str]) -> Union[int, str]:
    if isinstance(q, str) and len(q) == _HASH_LEN:
        # It's a hash
        return q
    chart_id_int = soft_int(q)
    if isinstance(chart_id_int, int):
        # It's an index
        return chart_id_int
    else:
        # It's neither...and probably bad
        return q
    

def dt2str(d: Dict) -> Dict:
    r = {}
    for k in d:
        if isinstance(d[k], dt):
            r[k] = d[k].isoformat()
        else:
            r[k] = d[k]
    return r


def soft_int(s: str) -> Union[int, str]:
    try:
        return int(s)
    except:
        return s

# Database connection
def engine_construct():
    url_construct = URL.create(
        "mssql+pyodbc",
        username=os.environ["SCOBILITY_UID"],
        password=os.environ["SCOBILITY_PWD"],
        host=os.environ["SCOBILITY_SERVER"],
        database=os.environ["SCOBILITY_DATABASE"],
        query={"driver": "ODBC Driver 18 for SQL Server"}
    )

    return create_engine(url_construct)

_ENGINE = engine_construct()
_MD = MetaData()
_MD.reflect(bind=_ENGINE)


_CATALOGS = {}
_SPICES = {}
_PLAYERS = {}
_SCORES = {}

## Database queries
# The catalog listing doesn't change often.
def _cache_catalogs() -> bool:
    if len(_CATALOGS) == 0:
        try:
            with _ENGINE.connect() as connection:
                result = connection.execute(
                    select(_MD.tables['Catalog'])
                )
                rows = result.all()
                for row in rows:
                    r = TypeAdapter(Catalog).validate_python(row._mapping)
                    _CATALOGS[r.name] = dt2str(dict(r))
            return True
        except Exception as e:
            logging.error(e)
            return False
    else:
        return True

# Spice ratings don't change often.
def _cache_spice(catalog_id: int) -> bool:
    if catalog_id not in _SPICES:
        try:
            # Need to query database and download/cache the spice ratings
            with _ENGINE.connect() as connection:
                result = connection.execute(
                    select(_MD.tables['Chart']).where(
                        _MD.tables['Chart'].c.catalog_id == catalog_id
                    )
                )
                rows = result.all()
                charts = []
                for row in rows:
                    charts.append(TypeAdapter(Chart).validate_python(row._mapping))
            _SPICES[catalog_id] = charts
            return True
        except Exception as e:
            logging.error(e)
            return False
    else:
        return True

# Player scobilities in the database are meant to be cached.
def _cache_players(catalog_id: int) -> bool:
    if catalog_id not in _PLAYERS:
        try:
            # Need to query database and download/cache the scobility info
            with _ENGINE.connect() as connection:
                result = connection.execute(
                    select(_MD.tables['Player']).where(
                        _MD.tables['Player'].c.catalog_id == catalog_id
                    )
                )
                rows = result.all()
                players = []
                for row in rows:
                    players.append(TypeAdapter(Player).validate_python(row._mapping))
            _PLAYERS[catalog_id] = {p.entrant_id: p for p in players}
            return True
        except Exception as e:
            logging.error(e)
            return False
    else:
        return True

# I don't want to also have to cache scores.
def _cache_scores(catalog_id: int, entrant_id: int) -> bool:
    if catalog_id not in _SCORES:
        _SCORES[catalog_id] = {}

    _cache_players(catalog_id)
    if entrant_id not in _PLAYERS[catalog_id]:
        logging.error(f"Couldn't find player #{entrant_id} in catalog #{catalog_id}")
        return False
    
    if entrant_id not in _SCORES[catalog_id]:
        try:
            player = _PLAYERS[catalog_id][entrant_id]
            performance_id = player.performance_id
            # Need to query database and download/cache the scores
            with _ENGINE.connect() as connection:
                result = connection.execute(
                    select(_MD.tables['Score']).where(
                        _MD.tables['Score'].c.catalog_id == catalog_id,
                        _MD.tables['Score'].c.performance_id == performance_id
                    )
                )
                rows = result.all()
                scores = []
                for row in rows:
                    scores.append(TypeAdapter(Score).validate_python(row._mapping))
            _SCORES[catalog_id][entrant_id] = scores
            return True
        except Exception as e:
            logging.error(e)
            return False
    else:
        return True


def _lookup_catalog(catalog_name: str) -> Dict:
    if not _cache_catalogs():
        return {
            'status': False,
            'message': f"Couldn't list scobility catalogs in DB"
        }

    if catalog_name in _CATALOGS:
        return {
            'status': True,
            'data': _CATALOGS[catalog_name],
            'message': f"Found {catalog_name} in scobility catalogs"
        }
    else:
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }


def _lookup_spice(catalog_name: str, chart_id: Union[int, str]) -> Dict:
    filter_fields = ['chart_id', 'hash', 'spice', 'spice_calc_time']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        if not _cache_spice(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get spice ratings for scobility catalog {catalog_name}"
            }

        cx = cid(chart_id)
        if isinstance(cx, str):
            # chart hash lookup
            chart_filter = [c for c in _SPICES[catalog_id] if c.hash == cx]
        else:
            # chart ID lookup
            chart_filter = [c for c in _SPICES[catalog_id] if c.chart_id == cx]
            
        if len(chart_filter) == 1:
            chart_selected = dt2str(dict(chart_filter[0]))
            return {
                'status': True,
                'data': {k: chart_selected[k] for k in chart_selected if k in filter_fields},
                'message': f"Found chart {chart_id} in scobility catalog {catalog_name}"
            }
        else:
            return {
                'status': False,
                'message': f"Couldn't find chart {chart_id} in scobility catalog {catalog_name}"
            }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }


def _list_all_spice(catalog_name: str, key: str = 'id') -> Dict:
    filter_fields = ['chart_id', 'hash', 'spice', 'spice_calc_time']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        if not _cache_spice(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get spice ratings for scobility catalog {catalog_name}"
            }

        if key.lower() == 'id':
            charts = {c.chart_id: dt2str(dict(c)) for c in _SPICES[catalog_id]}
        else:
            key = 'hash'
            charts = {c.hash: dt2str(dict(c)) for c in _SPICES[catalog_id]}
        charts_filtered = {k:
                {f: c[f] for f in c if f in filter_fields}
            for k, c in charts.items()}
            
        return {
            'status': True,
            'data': charts_filtered,
            'message': f"Listed all charts in scobility catalog {catalog_name} by {key}"
        }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }


def _lookup_player(catalog_name: str, entrant_id: int) -> Dict:
    filter_fields = ['entrant_id', 'scobility', 'timing_power', 'comfort_zone', 'scobility_calc_time']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        if not _cache_players(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get player data for scobility catalog {catalog_name}"
            }

        # entrant ID lookup
        if entrant_id in _PLAYERS[catalog_id]:
            player_selected = dt2str(dict(_PLAYERS[catalog_id][entrant_id]))
            return {
                'status': True,
                'data': {k: player_selected[k] for k in player_selected if k in filter_fields},
                'message': f"Found player {entrant_id} in scobility catalog {catalog_name}"
            }
        else:
            return {
                'status': False,
                'message': f"Couldn't find player {entrant_id} in scobility catalog {catalog_name}"
            }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }


def _list_all_players(catalog_name: str) -> Dict:
    filter_fields = ['entrant_id', 'scobility', 'timing_power', 'comfort_zone', 'scobility_calc_time']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        if not _cache_players(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get player data for scobility catalog {catalog_name}"
            }
            
        players = {k: dt2str(dict(p)) for k, p in _PLAYERS[catalog_id].items()}
        players_filtered = {k:
                {f: p[f] for f in p if f in filter_fields}
            for k, p in players.items()}
            
        return {
            'status': True,
            'data': players_filtered,
            'message': f"Listed all players in scobility catalog {catalog_name}"
        }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }


def _lookup_score(catalog_name: str, entrant_id: int, chart_id: Union[int, str]) -> Dict:
    filter_fields = ['score_id', 'entrant_id', 'chart_id', 'hash', 'score', 'plays', 'last_played']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        # player grab
        if not _cache_players(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get player data for scobility catalog {catalog_name}"
            }

        # player ID lookup
        if entrant_id not in _PLAYERS[catalog_id]:
            return {
                'status': False,
                'message': f"Couldn't find player {entrant_id} in scobility catalog {catalog_name}"
            }
        
        # chart grab
        if not _cache_spice(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get spice data for scobility catalog {catalog_name}"
            }
        
        # chart lookup
        cx = cid(chart_id)
        if isinstance(cx, str):
            # chart hash lookup
            chart_filter = [c for c in _SPICES[catalog_id] if c.hash == cx]
        else:
            # chart ID lookup
            chart_filter = [c for c in _SPICES[catalog_id] if c.chart_id == cx]
        if len(chart_filter) != 1:
            return {
                'status': False,
                'message': f"Couldn't find chart {chart_id} in scobility catalog {catalog_name}"
            }
        chart_selected = chart_filter[0]
        global_chart_id = chart_selected.global_chart_id

        # score grab
        if not _cache_scores(catalog_id, entrant_id):
            return {
                'status': False,
                'message': f"Couldn't get score data for player #{entrant_id} in scobility catalog {catalog_name}"
            }
        
        # score ID lookup
        score_filter = [s for s in _SCORES[catalog_id][entrant_id] if s.global_chart_id == global_chart_id]
        if len(score_filter) != 1:
            return {
                'status': True,
                'data': {
                    'score_id': -1,
                    'entrant_id': entrant_id,
                    'chart_id': chart_selected.chart_id,
                    'hash': chart_selected.hash,
                    'score': 0,
                    'plays': 0,
                    'last_played': dt.isoformat(dt.utcnow())
                },
                'message': f"Player {entrant_id} does not have a score on chart {chart_id} in scobility catalog {catalog_name}."
            }
        else:
            score_selected = dt2str(dict(score_filter[0]))
            score_data = {k: score_selected[k] for k in score_selected if k in filter_fields}
            score_data['entrant_id'] = entrant_id
            score_data['chart_id'] = chart_selected.chart_id
            score_data['hash'] = chart_selected.hash

            return {
                'status': True,
                'data': score_data,
                'message': f"Found score {score_data['score']} ({score_data['plays']} plays) for player {entrant_id} in scobility catalog {catalog_name}"
            }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }


def _list_all_scores(catalog_name: str, entrant_id: int) -> Dict:
    filter_fields = ['score_id', 'entrant_id', 'chart_id', 'hash', 'score', 'plays', 'last_played']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        # player grab
        if not _cache_players(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get player data for scobility catalog {catalog_name}"
            }

        # player ID lookup
        if entrant_id not in _PLAYERS[catalog_id]:
            return {
                'status': False,
                'message': f"Couldn't find player {entrant_id} in scobility catalog {catalog_name}"
            }
        
        # chart grab
        if not _cache_spice(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get spice data for scobility catalog {catalog_name}"
            }
        
        global_chart_id_list = {c.global_chart_id: c for c in _SPICES[catalog_id]}

        # score grab
        if not _cache_scores(catalog_id, entrant_id):
            return {
                'status': False,
                'message': f"Couldn't get score data for player #{entrant_id} in scobility catalog {catalog_name}"
            }
        
        # score ID lookup
        score_filter = [s for s in _SCORES[catalog_id][entrant_id] if s.global_chart_id in global_chart_id_list]

        scores = {}
        for s in score_filter:
            score_selected = dt2str(dict(s))
            score_data = {k: score_selected[k] for k in score_selected if k in filter_fields}
            score_data['entrant_id'] = entrant_id
            score_data['chart_id'] = global_chart_id_list[s.global_chart_id].chart_id
            score_data['hash'] = global_chart_id_list[s.global_chart_id].hash
            scores[score_data['hash']] = score_data

        return {
            'status': True,
            'data': scores,
            'message': f"Found {len(scores)} scores for player {entrant_id} in scobility catalog {catalog_name}"
        }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }
    

# Scobility logic surface
def calc_scobility(catalog_name: str, scores: Dict) -> Dict:
    # scores = {chart_id: score}

    scobility_result = {
        'entrant_id': None,
        'scobility': None,
        'timing_power': None,
        'comfort_zone': None,
        'scobility_calc_time': dt.isoformat(dt.utcnow()),
        'score_qualities': None,
        'message': None
    }
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        if not _cache_spice(catalog_id):
            return {
                'status': False,
                'message': f"Couldn't get spice ratings for scobility catalog {catalog_name}"
            }

        catalog = TypeAdapter(Catalog).validate_python(_CATALOGS[catalog_name])
        scores_dict = {soft_int(k): v for k, v in scores.items()}
        spices_dict = {c.chart_id: c.spice for c in _SPICES[catalog_id]}
        scobility_result = scobility_lite.calc_player_scobility(catalog, scores_dict, spices_dict)
        return {
            'status': True,
            'data': scobility_result,
            'message': scobility_result.get('message',  f"Calculated scobility catalog from {len(scores)} scores")
        }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }
    

def derive_target_score(catalog_name: str, entrant_info: Union[int, Dict], chart_q: Union[int, str]) -> Dict:    # entrant_info could be either:
    # - an entrant ID
    # - a dict of {chart_id: score}
    # chart_q could be either:
    # - a specific chart ID/hash
    # - "all", "id", or "hash", in which case calculate all the target scores for this player
    if isinstance(soft_int(entrant_info), int):
        result_player = _lookup_player(catalog_name, entrant_info)
        if 'data' in result_player:
            players = {result_player['data']['entrant_id']: result_player['data']}
    else:
        result_player = calc_scobility(catalog_name, entrant_info)
        if 'data' in result_player:
            players = result_player['data']
    if result_player['status'] == False:
        return {
            'status': False,
            'message': "Couldn't derive target score: " + result_player['message']
        }
        
    cx = cid(chart_q)
    if isinstance(cx, int):
        print(f"Looking for chart #{chart_q} in {catalog_name} catalog...")
        result_spices = _lookup_spice(catalog_name, cx)
        if 'data' in result_spices:
            spices = {result_spices['data']['chart_id']: result_spices['data']}
    elif chart_q.lower() in ["all", "id", "hash"]:
        print(f"Listing all charts in {catalog_name} catalog...")
        result_spices = _list_all_spice(catalog_name)
        if 'data' in result_spices:
            spices = result_spices['data']
    else:
        print(f"Looking for chart hash={chart_q} in {catalog_name} catalog...")
        result_spices = _lookup_spice(catalog_name, cx)
        if 'data' in result_spices:
            spices = {result_spices['data']['hash']: result_spices['data']}
    if result_spices['status'] == False:
        return {
            'status': False,
            'message': "Couldn't derive target score: " + result_spices['message']
        }
    
    if len(spices) * len(players) > _TOO_MANY_THINGS:
        return {
            'status': False,
            'message': f"Exceeded request limit (requested {len(players)} targets on {len(spices)} charts, limit is {_TOO_MANY_THINGS})"
        }
    
    catalog = TypeAdapter(Catalog).validate_python(_CATALOGS[catalog_name])
    
    target_scores = {}
    for p in players.values():
        player = Player(**p,
            performance_id=-1,
            catalog_id=-1,
            groovestats_id=-1,
            boogiestats_id=-1,
            name=""
            )
        targets = scobility_lite.calc_target_score(catalog, player, spices)
        target_scores[p['entrant_id']] = targets
    
    return {
        'status': True,
        'data': target_scores,
        'message': f"Derived target scores for {len(players)} players on {len(spices)} spice ratings"
    }


## API layer

# Heartbeat
@api.get("/")
def root() -> JSONResponse:
    s = "Scobility API! :chili_pepper:"
    print(s)
    return JSONResponse(content={'status': True, 'message': s})


# Retrieve info for a named scobility catalog
@api.get("/catalog/{catalog_name}")
def get_catalog(catalog_name: str) -> JSONResponse:
    print(f"Looking for {catalog_name} catalog...")

    result = _lookup_catalog(catalog_name)

    return JSONResponse(content=result,
        status_code=result.get('status') and 200 or 404)


# Retrieve spice rating for a particular chart ID or hash, or for all charts
@api.get("/catalog/{catalog_name}/chart/{chart_q}")
def get_chart(catalog_name: str, chart_q: str) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        cx = cid(chart_q)
        if isinstance(cx, int):
            print(f"Looking for chart #{cx} in {catalog_name} catalog...")
            result = _lookup_spice(catalog_name, cx)
        elif chart_q.lower() in ["all", "id", "hash"]:
            print(f"Listing all charts in {catalog_name} catalog...")
            result = _list_all_spice(catalog_name, chart_q.lower())
        else:
            print(f"Looking for chart hash={cx} in {catalog_name} catalog...")
            result = _lookup_spice(catalog_name, cx)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }


# Retrieve cached scobility for a specific entrant ID
@api.get("/catalog/{catalog_name}/scobility/{entrant_id}")
def get_scobility_cached(catalog_name: str, entrant_id: int) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Looking for player #{entrant_id} in {catalog_name} catalog...")
        result = _lookup_player(catalog_name, entrant_id)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }
    

# Retrieve cached scobility for all entrants in a catalog
@api.get("/catalog/{catalog_name}/scobility")
def get_scobility_all(catalog_name: str) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Listing scobility for all players in {catalog_name} catalog...")
        result = _list_all_players(catalog_name)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }
    

class ScobilityRequest(BaseModel):
    scores: Dict

# Calculate scobility for an arbitrary list of high scores
@api.post("/catalog/{catalog_name}/scobility")
def get_scobility_arbitrary(catalog_name: str, reqs: ScobilityRequest) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Calculating scobility for {len(reqs.scores)} scores provided in {catalog_name} catalog...")
        result = calc_scobility(catalog_name, reqs.scores)
        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }


# Calculate target score for a particular player,
# on a chart with a particular chart ID or hash - or for all charts
@api.get("/catalog/{catalog_name}/target/{entrant_id}/chart/{chart_q}")
def get_target_score(catalog_name: str, entrant_id: int, chart_q: str) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Targeting score for player #{entrant_id} on chart {chart_q} in {catalog_name} catalog...")
        result = derive_target_score(catalog_name, entrant_id, chart_q)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }


# Calculate target scores for a particular player on all charts
@api.get("/catalog/{catalog_name}/target/{entrant_id}")
def get_all_targets(catalog_name: str, entrant_id: int) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Targeting scores for player #{entrant_id} on all charts in {catalog_name} catalog...")
        result = derive_target_score(catalog_name, entrant_id, 'all')

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }


# Retrieve a player's score for a particular chart ID or hash
@api.get("/catalog/{catalog_name}/score/{entrant_id}/chart/{chart_q}")
def get_score(catalog_name: str, entrant_id: int, chart_q: str) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Looking for player #{entrant_id}'s score on chart {chart_q} in {catalog_name} catalog...")
        result = _lookup_score(catalog_name, entrant_id, chart_q)
            
        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }


# Retrieve a player's score for all charts in the catalog
@api.get("/catalog/{catalog_name}/score/{entrant_id}")
def get_all_scores(catalog_name: str, entrant_id: int) -> JSONResponse:
    result = _lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        print(f"Looking for player #{entrant_id}'s scores in {catalog_name} catalog...")
        result = _list_all_scores(catalog_name, entrant_id)
            
        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }
    

app = func.AsgiFunctionApp(
    app=api,
    http_auth_level=func.AuthLevel.ANONYMOUS
)
