from datetime import datetime as dt
from typing import Union

from pydantic import BaseModel


class Catalog(BaseModel):
    catalog_id: int
    name: str
    active: bool
    last_update: Union[dt, None]
    perfect_offset: Union[float, None]
    # perfect_offset = internal scobility parameter to prevent singularity at perfect scores
    perfect_score: Union[float, None]
    # perfect_score = 10000 (for ITG EX)
    #                 1000000 (for SMX)
    #                 None (if already expressed as proportional diff from perfect)


class Chart(BaseModel):
    global_chart_id: int
    catalog_id: int
    chart_id: int
    hash: str
    title: Union[str, None]
    subtitle: Union[str, None]
    artist: Union[str, None]
    meter: Union[float, None]
    slot: Union[str, None]
    style: Union[str, None]
    value: Union[float, None]
    spice: Union[float, None]
    spice_calc_time: Union[dt, None]


class Player(BaseModel):
    performance_id: int
    catalog_id: int
    entrant_id: int
    groovestats_id: int
    boogiestats_id: Union[int, None]
    name: Union[str, None]
    scobility: Union[float, None]
    timing_power: Union[float, None]
    comfort_zone: Union[float, None]
    scobility_calc_time: Union[dt, None]


class Relationship(BaseModel):
    relationship_id: int
    x_id: int
    y_id: int
    common: Union[int, None]
    relation: Union[float, None]
    strength: Union[float, None]


class Score(BaseModel):
    score_id: int
    catalog_id: int
    global_chart_id: int
    performance_id: int
    plays: Union[int, None]
    last_played: Union[dt, None]
    clear: Union[str, None]
    score: Union[float, None]
    prediction: Union[float, None]
    prediction_calc_time: Union[dt, None]
