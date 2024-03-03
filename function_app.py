import azure.functions as func

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


def dt2str(d: Dict) -> Dict:
    r = {}
    for k in d:
        if isinstance(d[k], dt):
            r[k] = d[k].isoformat()
        else:
            r[k] = d[k]
    return r

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

def enumerate_catalogs():
    catalogs = {}
    try:
        with _ENGINE.connect() as connection:
            result = connection.execute(
                select(_MD.tables['Catalog'])
            )
            rows = result.all()
            for row in rows:
                r = TypeAdapter(Catalog).validate_python(row._mapping)
                catalogs[r.name] = dt2str(dict(r))
            return catalogs
    except Exception as e:
        print(e)

_CATALOGS = enumerate_catalogs()


def lookup_catalog(catalog_name: str) -> Dict:
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

def lookup_spice(catalog_name: str, chart_id: Union[int, None]) -> Dict:
    filter_fields = ['chart_id', 'hash', 'spice', 'spice_calc_time']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        with _ENGINE.connect() as connection:
            if chart_id is None:
                result = connection.execute(
                    select(_MD.tables['Chart']).where(
                        _MD.tables['Chart'].c.catalog_id == catalog_id
                    )
                )
                rows = result.all()
                charts = []
                for row in rows:
                    c = TypeAdapter(Chart).validate_python(row._mapping)
                    cc = dt2str(dict(c))
                    charts.append({k: cc[k] for k in cc if k in filter_fields})
                return {
                    'status': True,
                    'data': charts,
                    'message': f"Found {len(charts)} charts in scobility catalog {catalog_name}"
                }
            else:
                if isinstance(chart_id, str):
                    # chart hash lookup
                    result = connection.execute(
                        select(_MD.tables['Chart']).where(
                            _MD.tables['Chart'].c.catalog_id == catalog_id,
                            _MD.tables['Chart'].c.hash == chart_id
                        )
                    )
                else:
                    # chart ID lookup
                    result = connection.execute(
                        select(_MD.tables['Chart']).where(
                            _MD.tables['Chart'].c.catalog_id == catalog_id,
                            _MD.tables['Chart'].c.chart_id == chart_id
                        )
                    )
                rows = result.all()
                if len(rows) == 1:
                    c = TypeAdapter(Chart).validate_python(rows[0]._mapping)
                    cc = dt2str(dict(c))
                    return {
                        'status': True,
                        'data': {k: cc[k] for k in cc if k in filter_fields},
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


def get_cached_scobility(catalog_name: str, entrant_id: Union[int, None]) -> Dict:
    filter_fields = ['entrant_id', 'scobility', 'timing_power', 'comfort_zone', 'scobility_calc_time']
    try:
        if catalog_name not in _CATALOGS:
            return {
                'status': False,
                'message': f"Couldn't find {catalog_name} in scobility catalogs"
            }
        else:
            catalog_id = _CATALOGS[catalog_name]['catalog_id']

        with _ENGINE.connect() as connection:
            if entrant_id is None:
                result = connection.execute(
                    select(_MD.tables['Player']).where(
                        _MD.tables['Player'].c.catalog_id == catalog_id
                    )
                )
                rows = result.all()
                players = []
                for row in rows:
                    p = TypeAdapter(Player).validate_python(row._mapping)
                    pp = dt2str(dict(p))
                    players.append({k: pp[k] for k in pp if k in filter_fields})
                return {
                    'status': True,
                    'data': players,
                    'message': f"Found {len(players)} players in scobility catalog {catalog_name}"
                }
            else:
                result = connection.execute(
                    select(_MD.tables['Player']).where(
                        _MD.tables['Player'].c.catalog_id == catalog_id,
                        _MD.tables['Player'].c.entrant_id == entrant_id
                    )
                )
                rows = result.all()
                if len(rows) == 1:
                    p = TypeAdapter(Player).validate_python(rows[0]._mapping)
                    pp = dt2str(dict(p))
                    return {
                        'status': True,
                        'data': {k: pp[k] for k in pp if k in filter_fields},
                        'message': f"Found player #{entrant_id} in scobility catalog {catalog_name}"
                    }
                else:
                    return {
                        'status': False,
                        'message': f"Couldn't find player #{entrant_id} in scobility catalog {catalog_name}"
                    }

    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Couldn't connect to DB"
        }


def calc_scobility(catalog_name: str, scores: Dict, rescale: Union[float, None] = None) -> Dict:
    # scores = {chart_id: score_ex_unity}
    # rescale = 10000 (for ITG EX)
    #           1000000 (for SMX)
    #           None (if already expressed as proportional diff from perfect)

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

        with _ENGINE.connect() as connection:
            result = connection.execute(
                select(_MD.tables['Chart']).where(
                    _MD.tables['Chart'].c.catalog_id == catalog_id,
                    # _MD.tables['Chart'].c.chart_id in scores.keys()
                )
            )
            charts = [
                TypeAdapter(Chart).validate_python(r._mapping)
                for r in result.all()
            ]
            scores_rescale = scores
            if rescale is not None:
                scores_rescale = {
                    int(k): 1 - v / rescale for k, v in scores.items()
                }
            scobility_result = scobility_lite.calc_player_scobility(
                scores_rescale,
                {c.chart_id: c.spice for c in charts}
            )
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



@api.get("/")
def root() -> JSONResponse:
    s = "Scobility API! :chili_pepper:"
    print(s)
    return JSONResponse(content={'status': True, 'message': s})


@api.get("/catalog/{catalog_name}")
def get_catalog(catalog_name: str) -> JSONResponse:
    print(f"Looking for {catalog_name} catalog...")

    result = lookup_catalog(catalog_name)

    return JSONResponse(content=result,
        status_code=result.get('status') and 200 or 404)


@api.get("/catalog/{catalog_name}/chart/{chart_q}")
def get_chart(catalog_name: str, chart_q: str) -> JSONResponse:
    result = lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        chart_id = int(chart_q)
        print(f"Looking for chart #{chart_id} in {catalog_name} catalog...")
        result = lookup_spice(catalog_name, chart_id)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except ValueError:
        if chart_q.lower() == "all":
            print(f"Listing all charts in {catalog_name} catalog...")
            result = lookup_spice(catalog_name, None)
        else:
            print(f"Looking for chart hash={chart_q} in {catalog_name} catalog...")
            result = lookup_spice(catalog_name, chart_q)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }


@api.get("/catalog/{catalog_name}/scobility/{player_q}")
def get_scobility_cached(catalog_name: str, player_q: str) -> JSONResponse:
    result = lookup_catalog(catalog_name)
    if 'data' not in result:    
        return {
            'status': False,
            'message': f"Couldn't find {catalog_name} in scobility catalogs"
        }
    
    try:
        player_id = int(player_q)
        print(f"Looking for player #{player_id} in {catalog_name} catalog...")
        result = get_cached_scobility(catalog_name, player_id)

        return JSONResponse(content=result,
            status_code=result.get('status') and 200 or 404)
    except ValueError:
        if player_q.lower() == "all":
            print(f"Listing scobility for all players in {catalog_name} catalog...")
            result = get_cached_scobility(catalog_name, None)
            return JSONResponse(content=result,
                status_code=result.get('status') and 200 or 404)
        else:
            return {
                'status': False,
                'message': "Something unexpected happened"
            }
    except Exception as e:
        print(e)
        return {
            'status': False,
            'message': "Something unexpected happened"
        }
    

class ScobilityRequest(BaseModel):
    scores: Dict
    rescale: Union[float, None] = None

@api.post("/catalog/{catalog_name}/scobility")
def get_scobility(catalog_name: str, reqs: ScobilityRequest) -> JSONResponse:
    try:
        print(f"Calculating scobility for {len(reqs.scores)} scores provided in {catalog_name} catalog...")
        result = calc_scobility(catalog_name, reqs.scores, reqs.rescale)
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
