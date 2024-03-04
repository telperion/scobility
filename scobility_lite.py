import os
from datetime import datetime as dt, timezone as tz
from typing import Union, Dict

import numpy as np
from pydantic import BaseModel, TypeAdapter

from db_types import Catalog, Chart, Player, Relationship, Score

_MIN_CHARTS_SCOBILITY_CALC = 5
_RANKING_CHART_COUNT = 75

def calc_score_quality(catalog: Catalog, scores: np.ndarray, spices: np.ndarray):
    scores_rescaled = scores
    if catalog.perfect_score is not None:
        scores_rescaled = 1 - scores/catalog.perfect_score
    # The lowest possible score (0%) on the chart with
    # the lowest spice level will define the (0, 0) point.
    p_min = np.log2(catalog.perfect_offset + 1)
    p_max = np.log2(catalog.perfect_offset)
    p_spices = np.log2(spices)
    p_scores = np.log2(catalog.perfect_offset + scores_rescaled) - p_min

    return (p_spices - p_scores)


def calc_player_scobility(catalog: Catalog, scores_dict: Dict, spices_dict: Dict):
    scobility_result = {
        'entrant_id': None,
        'scobility': None,
        'timing_power': None,
        'comfort_zone': None,
        'scobility_calc_time': dt.isoformat(dt.utcnow()),
        'score_qualities': None,
        'message': None
    }

    factor_charts = list(set(scores_dict.keys()) & set(spices_dict.keys()))

    # If you don't have enough plays on charts with spice rankings,
    # you don't get a scobility rating. Sorry.
    if len(factor_charts) < _MIN_CHARTS_SCOBILITY_CALC:
        scobility_result['message'] = f"Not enough scores on spiced charts to calculate scobility (needed {_MIN_CHARTS_SCOBILITY_CALC}, got {len(factor_charts)})"
        return scobility_result
    
    scores = np.array([scores_dict[k] for k in factor_charts])
    spices = np.array([spices_dict[k] for k in factor_charts])

    p_quality = calc_score_quality(catalog, scores, spices)

    # Draw a best-fit line to calculate the player's strength.
    # Mostly used to determine the "comfort zone", i.e.
    # where do you outperform your peers? easier or harder charts?
    # by observing the slope of the best-fit line.
    # TODO: come up with a better-fitting equation that accounts for
    # the "struggle knee" (piecewise linear?)
    a = np.log2(spices)
    b = p_quality         # Expected to be constant...
    w = a # np.ones_like(a) # 1 - 1/(a + Relationship.WEIGHT_OFFSET)**2
    m = np.vstack([np.ones_like(a), a])
    coefs = np.linalg.lstsq(m.T * w[:, np.newaxis], b * w, rcond=None)[0]
    b_eq = coefs @ m

    # TODO: derive a volforce-like "tournament power" that rewards
    # playing more songs as well as getting better scores
    # this is a pretty silly first stab at it imo
    # player.tourney_power = sum(p_quality) / len(p_quality) * np.sqrt(len(p_quality) / len(self.ordering)) * Tournament.MAX_TOURNEY_POWER
    scobility_result['scobility'] = sum(p_quality) / len(p_quality)  # Simple average...
    scobility_result['timing_power'] = coefs[0]
    scobility_result['comfort_zone'] = coefs[1]
    scobility_result['score_qualities'] = {k: v for k, v in zip(factor_charts, p_quality)}
    scobility_result['message'] = f"Calculated scobility from scores on {len(scores_dict)} & {len(spices_dict)} = {len(factor_charts)} spiced charts"

    return scobility_result


def calc_target_score(catalog: Catalog, player: Player, spices: np.ndarray) -> Union[np.ndarray, None]:
    if (player.timing_power is None) or (player.comfort_zone is None):
        return None
    
    # The lowest possible score (0%) on the chart with
    # the lowest spice level will define the (0, 0) point.
    p_min = np.log2(catalog.perfect_offset + 1)
    p_max = np.log2(catalog.perfect_offset)
    p_spices = np.log2(spices)
    #p_scores = np.log2(_PERFECT_OFFSET + scores) - p_min

    # Score quality that would bring this chart up to the player's scobility fit
    predicted_quality = player.timing_power + p_spices * player.comfort_zone

    # Missing %EX score that would bring this chart up to the player's scobility fit
    ex_target = np.clip((np.power(2, p_spices - predicted_quality)) * (catalog.perfect_offset + 1), a_min=0, a_max=1)

    if catalog.perfect_score is not None:
        return catalog.perfect_score * (1 - ex_target)
    else:
        return ex_target


def calc_point_curve(catalog: Catalog, v_raw: np.ndarray) -> np.ndarray:
    v = 100 - 100*v_raw

    if catalog.name == 'ITL2023':
        log_base = 1.1032889141348
        pow_base = 61
        inflect = 50
    elif catalog.name == 'ITL2022':
        log_base = 1.0638215
        pow_base = 31
        inflect = 75
    else:
        return np.zeros_like(v)

    v_lo = np.clip(v, a_min=None, a_max=inflect)
    v_hi = np.clip(v, a_min=inflect, a_max=None)
    
    return \
        np.log(v_lo + 1) / np.log(log_base) + \
        np.power(pow_base, (v_hi-inflect)/(100-inflect)) - 1


def calc_point_curve_inv(catalog: Catalog, p: np.ndarray) -> np.ndarray:
    if catalog.name == 'ITL2023':
        log_base = 1.1032889141348
        pow_base = 61
        inflect = 50
    elif catalog.name == 'ITL2022':
        log_base = 1.0638215
        pow_base = 31
        inflect = 75
    else:
        return np.zeros_like(p)

    piecewise_border = np.round(np.log(inflect + 1)/np.log(log_base) - 1, decimals=3)
    
    v = np.zeros_like(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        v[p <= piecewise_border] = np.power(log_base, p[p <= piecewise_border]) - 1
        v[p >  piecewise_border] = (100-inflect)*np.log(p[p > piecewise_border] - piecewise_border)/np.log(pow_base) + inflect
    return 1 - 0.01*v
