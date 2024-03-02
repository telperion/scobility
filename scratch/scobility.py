"""Scobility! :chili_pepper:"""

import json
import glob
import os
import re
import sys
import gc
import traceback as tb
import unicodedata
from io import StringIO
from functools import reduce
from warnings import warn
from datetime import datetime as dt, timedelta as td

import numpy as np
from matplotlib import pyplot as plt
import hsluv

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List

_VERSION = 'v0.973'
_VERBAL = False
_VISUAL = False


def slugify(value, allow_unicode=False):
    """
    https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value)
    return re.sub(r'[-\s]+', '-', value).strip('-_')

class Clear(IntEnum):
    FAIL = 0
    PASS = 1
    BATTERY = 2
    FC = 3
    FEC = 4
    QUAD = 5
    QUINT = 6

@dataclass
class Score:
    SCORE_SCALAR = 0.0001

    s_id: int = -1
    e_id: int = -1
    plays: int = 1
    last_played: dt = None
    clear: Clear = Clear.PASS
    value: float = 0
    
    def __init__(self, data):
        last_played = data.get('last_played')
        if isinstance(last_played, str):
            last_played = dt.strptime(last_played, '%Y-%m-%dT%H:%M:%S.%f')
        self.s_id = data['s_id']
        self.e_id = data['e_id']
        self.plays = data.get('plays', 1)
        self.last_played = last_played
        self.clear = Clear(data['clear'])
        self.value = data['value']

    def dump(self):
        return {
            's_id': self.s_id,
            'e_id': self.e_id,
            'plays': self.plays,
            'last_played': self.last_played and self.last_played.strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'clear': int(self.clear),
            'value': self.value
        }
    
    @classmethod
    def load(cls, data):
        return cls(data)

    @staticmethod
    def det_clear(data: dict):
        # Assume pass (because that's the only type of score that gets logged by ITL).
        c = Clear.PASS

        # FC is the most complicated to check.
        if (
            (data['score_mines_hit'] == 0) and
            (data['score_holds_held'] == data['song_total_holds']) and
            (data['score_rolls_held'] == data['song_total_rolls']) and
            (data['score_miss'] == 0) and
            ((data['score_w6'] is None) or data['score_w6'] == 0) and
            ((data['score_w5'] is None) or data['score_w5'] == 0)
        ):
            c = Clear.FC
        else:
            return c
        
        # FEC?
        if data['score_w4'] == 0:
            c = Clear.FEC
        else:
            return c

        # FFC (i.e., quad)?
        if data['score_w3'] == 0:
            c = Clear.QUAD
        else:
            return c

        # FBC (i.e., quint)?
        if data['score_w2'] == 0:
            c = Clear.QUINT
        return c


@dataclass
class Song:
    s_id: int = -1
    hash: str = None
    title: str = ''
    subtitle: str = ''
    artist: str = ''
    meter: float = 0
    slot: str = 'Challenge'
    value: float = 0                                # if the tourney specifies varying point values for charts
    scores: dict = field(default_factory=dict)      # e_id: Score
    spice: float = None                             # Not a Dune reference. capsaicin not cinnamon

    def __init__(self, data: dict = None):
        if data is None:
            return

        try:
            self.s_id = data['song_id']
            self.hash = data['song_hash']
            self.value = data['song_points']

            r = data['song_title_romaji'].strip()
            self.title = data['song_title'] + ((r != '') and f' ({r})' or '')
            
            r = data['song_subtitle_romaji'].strip()
            self.subtitle = data['song_subtitle'] + ((r != '') and f' ({r})' or '')

            r = data['song_artist_romaji'].strip()
            self.artist = data['song_artist'] + ((r != '') and f' ({r})' or '')

            self.meter = data['song_meter']
            self.slot = data['song_difficulty']
            self.playstyle = 1      # SP only
        except:
            self.s_id = data['id']
            self.hash = data['hash']
            self.value = data['points']

            r = data['titleRomaji'].strip()
            self.title = data['title'] + ((r != '') and f' ({r})' or '')
            
            r = data['subtitleRomaji'].strip()
            self.subtitle = data['subtitle'] + ((r != '') and f' ({r})' or '')

            r = data['artistRomaji'].strip()
            self.artist = data['artist'] + ((r != '') and f' ({r})' or '')

            self.meter = data['meter']
            self.slot = data['difficulty']
            self.playstyle = data['playstyle']

        self.scores = {}
        self.spice = None

    def dump(self):
        return {
            's_id': self.s_id,
            'hash': self.hash,
            'title': self.title,
            'subtitle': self.subtitle,
            'artist': self.artist,
            'meter': self.meter,
            'slot': self.slot,
            'value': self.value,
            'spice': self.spice
        }

    @classmethod
    def load(cls, data):
        obj = cls()
        for fn in ['s_id', 'hash', 'title', 'subtitle', 'artist', 'meter', 'slot', 'value', 'spice']:
            setattr(obj, fn, data[fn])
        return obj

    def __str__(self):
        return f"#{self.s_id} {self.full_title} ({self.slot} {self.meter}) ({self.value} max pts.)"

    @property
    def full_title(self):
        if len(self.subtitle) > 0:
            return f'{self.title} {self.subtitle}'
        else:
            return f'{self.title}'

    @property
    def name(self):
        return f"#{self.s_id} {self.full_title} ({self.slot} {self.meter})"


@dataclass
class Player:
    name: str
    e_id: int = -1
    g_id: int = -1
    scores: dict = field(default_factory=dict)      # s_id: Score

    scobility: float = None
    comfort_zone: float = None
    timing_power: float = None
    tourney_power: float = None

    def __init__(self, data: dict = None):
        if data is None:
            return

        try:
            self.name = data['members_name']
            self.e_id = data['entrant_id']
            self.g_id = data['entrant_members_id']
        except:
            self.name = data['name']
            self.e_id = data['id']
            self.g_id = data['membersId']
            
        self.scores = {}
        self.scobility = None
        self.comfort_zone = None
        self.timing_power = None
        self.tourney_power = None

    def dump(self):
        return {
            'name': self.name,
            'e_id': self.e_id,
            'g_id': self.g_id,
            'scobility': self.scobility,
            'comfort_zone': self.comfort_zone,
            'timing_power': self.timing_power,
            'tourney_power': self.tourney_power
        }

    @classmethod
    def load(cls, data):
        obj = cls()
        for fn in ['name', 'e_id', 'g_id', 'scobility', 'comfort_zone', 'timing_power', 'tourney_power']:
            setattr(obj, fn, data[fn])
        obj.scores = data.get('scores', {})
        return obj

    def __str__(self):
        return f"{self.name} (#{self.e_id})"


class Relationship:
    MAX_NEG_LIMIT = 0.3             # i.e., 700,000 min EX score
    MIN_NEG_LIMIT = 0.0002          # i.e., 999,800 max EX score (ONLY for initial score scaling!)
    WEIGHT_OFFSET = 0.5
    MIN_COMMON_PLAYERS = 2
    
    def __init__(self, x: Song, y: Song):
        self.x = x
        self.y = y
        x_players = set(x.scores)
        y_players = set(y.scores)
        self.e_common = x_players.intersection(y_players)

        self.relation = None
        self.strength = None
        self.use_link = None

    def dump(self):
        return {
            'x_id': self.x.s_id,
            'y_id': self.y.s_id,
            'relation': self.relation,
            'strength': self.strength
        }

    @classmethod
    def load(cls, data: dict, songs: dict):
        obj = cls(
            songs[data['x_id']],
            songs[data['y_id']]
        )
        obj.relation = data['relation']
        obj.strength = data['strength']
        return obj


    def pair_title(self):
        return f"{self.x} & {self.y}"

    def compare_title(self):
        return f"{self.y} vs. {self.x}"

    def __str__(self):
        return f"({self.compare_title()}): {self.relation:0.6f} (Strength: {self.strength:0.3f}" + (self.use_link and f", linked through {self.use_link})" or ")")

    def __repr__(self):
        if self.relation is None:
            return f"#{self.y.s_id} vs. #{self.x.s_id}: no relationship known"
        else:
            return f"#{self.y.s_id} vs. #{self.x.s_id}: {self.relation:0.6f} (strength {self.strength:0.3f}" + (self.use_link and f", link {self.use_link.s_id})" or ")")


    def calc_relationship(self):
        """
        Use the scores of players that have played both songs and
        perform some sort of fit to that data.

        It seems like, after snipping out the very top end (> 99%)
        and the lower end (< 80%) of scores...a linear fit kinda
        does the job well enough??

        I do have a weighting in so that better scores within that
        range have more influence on the best-fit calculation.
        """
        x_players = set(self.x.scores)
        y_players = set(self.y.scores)
        self.e_common = x_players.intersection(y_players)

        if len(self.e_common) < Relationship.MIN_COMMON_PLAYERS:
            raise ValueError(f'Not enough players to relate {self.compare_title()} ({len(self.e_common)} players, need {Relationship.MIN_COMMON_PLAYERS})')

        x_scores = [self.x.scores[e_id].value for e_id in self.e_common]
        y_scores = [self.y.scores[e_id].value for e_id in self.e_common]

        ex_matrix = np.vstack([x_scores, y_scores])

        screen_max_neg = np.amax(ex_matrix, axis=0)
        screen_min_neg = np.amin(ex_matrix, axis=0)

        ex_matrix_lfit1 = ex_matrix[:, np.logical_and(
            screen_max_neg < Relationship.MAX_NEG_LIMIT,
            screen_min_neg > Relationship.MIN_NEG_LIMIT
        )]
        ex_matrix_lfit2 = ex_matrix[:, 
            screen_max_neg < Relationship.MIN_NEG_LIMIT
        ]
        ex_matrix_screen = ex_matrix[:,
            screen_max_neg < Relationship.MAX_NEG_LIMIT
        ]

        x_col = ex_matrix_lfit1[0, :]
        y_col = ex_matrix_lfit1[1, :]

        if ex_matrix_lfit1.shape[1] < Relationship.MIN_COMMON_PLAYERS:
            min_score_check = (1 - Relationship.MAX_NEG_LIMIT) / Score.SCORE_SCALAR
            max_score_check = (1 - Relationship.MIN_NEG_LIMIT) / Score.SCORE_SCALAR
            raise ValueError(f'Not enough scores between {min_score_check:0.0f} and {max_score_check:0.0f} to relate {self.compare_title()} ({len(self.e_common)} players, need {Relationship.MIN_COMMON_PLAYERS})')
        
        # slope of line thru 0 as a starting point?
        a = x_col
        b = y_col
        w = 1 - 1/(a + Relationship.WEIGHT_OFFSET)**2
        p = np.vstack([a])
        coefs, resid = np.linalg.lstsq(p.T * w[:, np.newaxis], b * w, rcond=None)[:2]

        self.relation = coefs[0]                        # Slope of the best-fit line
        self.strength = np.sqrt(len(a) / resid[0])      # Reciprocal of residual, accounting for number of data points
        
        if _VERBAL:
            print(self)

        if _VISUAL and self.x.s_id == 280 and self.y.s_id == 430:
            p_sort = np.sort(a)[np.newaxis]
            b_eq = coefs @ p_sort

            plt.subplots(figsize=(6, 6))
            plt.plot(p_sort.T, b_eq, color='tab:blue')

            x_all = ex_matrix_screen[0, :]
            y_all = ex_matrix_screen[1, :]

            plt.scatter(x_all, y_all, color='tab:pink')
            plt.xlabel(f'{self.x}, Difference from 100%')
            plt.ylabel(f'{self.y}, Difference from 100%')
            plt.title(f'{self.compare_title()}\nRelation: {self.relation:0.3f}, Strength: {self.strength:0.3f}')
            plt.show()
            plt.close('all')
        


@dataclass
class Tournament:
    MONO_THRESHOLD = 0.999999                       # Monotonicity check
    MIN_COMMON_PLAYERS = 2                          # For ordering purposes
    ITERATIONS_MONOTONIC_SORT = 10                  # Bubble sort for correlation factor monotonicity
    ITERATIONS_SCOBILITY_SORT = 500                 # Refining the scobility values and post-sorting
    SCOBILITY_WINDOW_LOWER = 10                     # Incorporate this many lower scobility readings
    SCOBILITY_WINDOW_UPPER = 6                      # Incorporate this many higher scobility readings
    PERFECT_OFFSET = 0.003                          # Push scores away from logarithmic asymptote
    MAX_TOURNEY_POWER = 100                         # idk, I need a value
    RANKING_CHART_COUNT = 75                        # Only the top N of charts are considered for tourney ranking points
    POINT_CURVE = 'itl2023'                         # Function that converts %EX to % of points earned
    MAX_HEADTAIL_QUALITY = 5                        # Maximum number of highest/lowest score quality callouts (N highest & N lowest)
    MAX_RP_RECOMMENDATIONS = 10                     # Maximum number of tourney ranking point recovery recommendations to return

    players: dict = field(default_factory=dict)         # e_id: Player
    songs: dict = field(default_factory=dict)           # s_id: Song
    relationships: dict = field(default_factory=dict)   # r.x: {r.y: Relationship}
    ordering: list = field(default_factory=list)        # List[Song]

    def dump(self):
        j = {
            'songs': [],
            'players': [],
            'scores': [],
            'relationships': [],
            'ordering': []
        }
        for s in self.songs.values():
            j['songs'].append(s.dump())
            for v in s.scores.values():
                j['scores'].append(v.dump())
        for p in self.players.values():
            j['players'].append(p.dump())
        for x, y_dict in self.relationships.items():
            for y, r in y_dict.items():
                if r.relation is not None:
                    j['relationships'].append(r.dump())
        j['ordering'] = [s.s_id for s in self.ordering]

        return j

    @classmethod
    def load(cls, data):
        obj = cls()
        for s_data in data['songs']:
            s = Song.load(s_data)
            obj.songs[s.s_id] = s
        for p_data in data['players']:
            p = Player.load(p_data)
            obj.players[p.e_id] = p
        for v_data in data['scores']:
            v = Score.load(v_data)
            if v.s_id in obj.songs:
                if hasattr(obj.songs[v.s_id], 'scores'):
                    obj.songs[v.s_id].scores[v.e_id] = v
                else:
                    obj.songs[v.s_id].scores = {v.e_id: v}
            if v.e_id in obj.players:
                if hasattr(obj.players[v.e_id], 'scores'):
                    obj.players[v.e_id].scores[v.s_id] = v
                else:
                    obj.players[v.e_id].scores = {v.s_id: v}
        for r_data in data['relationships']:
            r = Relationship.load(r_data, obj.songs)
            if r.x.s_id not in obj.relationships:
                obj.relationships[r.x.s_id] = {}
            obj.relationships[r.x.s_id][r.y.s_id] = r
        obj.ordering = [obj.songs[s] for s in data['ordering']]
        return obj

    @staticmethod
    def link_through(a: Relationship, b: Relationship) -> Relationship:
        # Why not?
        if a.y.s_id != b.x.s_id:
            raise ValueError(f'Discontinuous link {a.y.s_id} -> {b.x.s_id}')

        r = Relationship(a.x, b.y)
        r.e_common = a.e_common.intersection(b.e_common)
        r.relation = a.relation * b.relation
        r.strength = (a.relation + b.relation) / (a.relation * b.strength + b.relation * a.strength)
        r.use_link = a.y

        return r

    @staticmethod
    def link_strength(links: List[Relationship]) -> Relationship:
        # Okay maybe this is a stretch
        if len(links) == 0:
            return None
        if len(links) == 1:
            return links[0]
        
        return reduce(Tournament.link_through, links)


    @staticmethod
    def colorize_dates(dates: List[float], alpha: int = 210) -> List[str]:
        dates_days = [round(d / 86400) for d in dates]
        dates_unique = sorted(list(set(dates_days)))
        colors_ref = [hsluv.hsluv_to_hex([345, 100 * saturation, 60 * np.sqrt(saturation)]) + f'{alpha:02x}' for saturation in np.linspace(1, 0, len(dates_unique))[::-1]]
        dates_match = [dates_unique.index(v) for v in dates_days]
        colors = [colors_ref[i] for i in dates_match]
        return colors


    def load_score_data(self, score_data: list, assign_to_player: bool = True):
        for s_data in score_data:
            try:
                s = Score(data={
                    's_id': s_data['song_id'],
                    'e_id': s_data['entrant_id'],
                    'clear': Clear(s_data['score_best_clear_type']),
                    'value': 1 - s_data['score_ex'] * Score.SCORE_SCALAR
                })
            except:
                if 'lastUpdated' in s_data:
                    s_dt = dt.strptime(s_data['lastUpdated'], '%Y-%m-%dT%H:%M:%S.%fZ')
                else:
                    s_dt = None
                s = Score(data={
                    's_id': s_data['chartId'],
                    'e_id': s_data['entrantId'],
                    'plays': s_data.get('totalPasses', 1),
                    'last_played': s_dt,
                    'clear': Clear(s_data.get('clearType', 1)),
                    'value': 1 - s_data['ex'] * Score.SCORE_SCALAR
                })

            if s.s_id in self.songs:
                self.songs[s.s_id].scores[s.e_id] = s
            if assign_to_player and (s.e_id in self.players):
                self.players[s.e_id].scores[s.s_id] = s

    def setup_relationships(self):
        for x in self.songs:
            prev = [y for y in self.relationships]
            self.relationships[x] = {}
            for y in prev:
                self.relationships[x][y] = Relationship(self.songs[x], self.songs[y])
                self.relationships[y][x] = Relationship(self.songs[y], self.songs[x])

    def calc_relationships(self, verbal=_VERBAL):
        successes = 0
        rel_list = [r for x, y_dict in self.relationships.items() for y, r in y_dict.items()]
        for i, r in enumerate(rel_list):
            try:
                r.calc_relationship()
                if verbal:
                    print(f'--- {i:6d} {r}')
                successes += 1
            except ValueError as e:
                print(f'!!! {i:6d} ', end='')
                print(e)
        
        print(f'Completed relationship calculations ({successes} out of {len(rel_list)} strong pairs)')


    def calc_relationship_safe(self, index: int, a: Song, b: Song, verbal=_VERBAL):
        try:
            r = self.relationships[a][b]
            r.calc_relationship()
            if verbal:
                print(f'--- {index:6d} {r}')
            return True
        except ValueError as e:
            print(f'!!! {index:6d} ', end='')
            print(e)
            return False


    def calc_relationships_jit(self, verbal=_VERBAL,
        src = 'song_scores',
        chunk_size = 100,
        exclude_diagonal = True,
        explicit_pairs = False
        ):
        # Memory-saving modification - only load a couple
        # score database partitions at one time.
        # (Also try to minimize loads!)

        # Partition the list of charts into chunks.
        song_id_list = [s for s in self.songs.keys()]
        songs_partition = [song_id_list[i:i+chunk_size] for i in range(0, len(self.songs), chunk_size)]
        rel_within_part = [False for s_part in songs_partition]

        # Generate the loading order by walking back and forth in
        # one triangular half of all possible pairs.
        load_replacement = []
        exc_diag = exclude_diagonal and 1 or 0
        for i_row in range(0, len(songs_partition), 2):
            # (i, i) --> (i, n)
            for i_col in range(i_row + exc_diag, len(songs_partition)):
                if explicit_pairs or (i_col == i_row + exc_diag):
                    load_replacement.append( (i_row, i_col) )
                else:
                    load_replacement.append( (None, i_col) )

            # (i+1, i+1) <-- (i+1, n)
            if i_row + 1 >= len(songs_partition):
                break
            for i_col in range(len(songs_partition) - 1, i_row + exc_diag, -1):
                if i_col == len(songs_partition) - 1:
                    load_replacement.append( (i_row + 1, explicit_pairs and i_col or None) )
                else:
                    load_replacement.append( (explicit_pairs and (i_row + 1) or None, i_col) )

        # print(load_replacement)

        # Calculate all the relationships!
        successes = 0
        attempted = 0
        index_a = -1
        index_b = -1
        part_a = []
        part_b = []
        for (repl_a, repl_b) in load_replacement:
            # Clear out memory used for the previous partitions.
            repl_load = []
            if repl_a is not None:
                for s_id in part_a:
                    del self.songs[s_id].scores
                    self.songs[s_id].scores = {}
                index_a = repl_a
                part_a = songs_partition[repl_a]
                repl_load += part_a
            if repl_b is not None:
                for s_id in part_b:
                    del self.songs[s_id].scores
                    self.songs[s_id].scores = {}
                index_b = repl_b
                part_b = songs_partition[repl_b]
                repl_load += part_b
            gc_unreachable = gc.collect()
            print(f'gc: {gc_unreachable}')

            if True: # verbal:
                print(f'@@@ Partitions: {index_a:4d} & {index_b:4d}')

            # Load scores for any charts that appear in fresh partitions.
            for s_id in repl_load:
                s = self.songs[s_id]
                # Try loading by song hash, then by song ID...
                fn = os.path.join(src, f'{s.hash}.json')
                if not os.path.exists(fn):
                    fn = os.path.join(src, f'{s_id}.json')
                if not os.path.exists(fn):
                    warn(f'Couldn\'t open a score data file matching {s.s_id} or {s.hash}', RuntimeWarning)

                with open(fn, 'r') as fp:
                    score_data = json.load(fp)
                    self.load_score_data(score_data['scores'], assign_to_player=False)

            # Calculate all relationships between these two partitions.
            for a in part_a:
                for b in part_b:
                    attempted += 1
                    successes += self.calc_relationship_safe(attempted, a, b, verbal) and 1 or 0
                    
                    attempted += 1
                    successes += self.calc_relationship_safe(attempted, b, a, verbal) and 1 or 0
                
            # Calculate all relationships within each partition
            # (if not done yet).
            for index_x, part_x in {index_a: part_a, index_b: part_b}.items():
                if not rel_within_part[index_x]:
                    for a in part_x:
                        for b in part_x:
                            if a != b:
                                attempted += 1
                                successes += self.calc_relationship_safe(attempted, a, b, verbal) and 1 or 0
                                
                                attempted += 1
                                successes += self.calc_relationship_safe(attempted, b, a, verbal) and 1 or 0
                    rel_within_part[index_x] = True

        
        print(f'Completed relationship calculations ({successes} out of {attempted} strong pairs)')

    def relationship_lookup(self, x: Song, y: Song, allow_link: bool = False) -> Relationship:
        # TODO: Oh no my data model
        if x.s_id == y.s_id:
            # Same song!
            r = Relationship(x, x)
            r.relation = 1.0
            r.strength = len(r.e_common)
            return r
        if x.s_id not in self.relationships:
            raise ValueError(f'No relationships originating from {x}')
        if y.s_id not in self.relationships[x.s_id]:
            if not allow_link:
                raise ValueError(f'No relationship for {y} vs. {x}')
            # Allow transitive relationship extension...
            potential_links = []
            for z_id in self.relationships[x.s_id]:
                if y.s_id in self.relationships[z_id]:
                    potential_links.append(Tournament.link_through(self.relationships[x.s_id][z_id], self.relationships[z_id][y.s_id]))
            if len(potential_links) < 1:
                raise ValueError(f'No relationship for {y} vs. {x}, even with linking!')
            potential_links.sort(key=lambda r: r.strength)
            return potential_links[-1]

        return self.relationships[x.s_id][y.s_id]

    def view_monotonicity(self, verbal=_VERBAL, visual=_VISUAL):
        order_progressive_rel = []
        for i_song in range(len(self.ordering)-1):
            order_progressive_rel.append(self.relationship_lookup(self.ordering[i_song], self.ordering[i_song+1]))

        if verbal:
            for r in order_progressive_rel:
                print(f"{r.pair_title():50s}:\n>>> corr. {r.relation:0.6f} from {len(r.e_common)} players")
                
        if visual:
            plt.plot([r.relation for r in order_progressive_rel])
            plt.plot([1 for r in order_progressive_rel])
            plt.show()

    def order_songs_initial(self, verbal=_VERBAL, visual=_VISUAL):
        self.ordering = []

        # Sort the relationships.
        # Only use upward-directional pairs
        # (i.e., second song is harder than the first)
        song_similarity = [r for x, y_dict in self.relationships.items() for y, r in y_dict.items() if
            (r.relation is not None) and
            (len(r.e_common) >= Tournament.MIN_COMMON_PLAYERS)
        ]
        song_similarity.sort(key=lambda x: x.relation)

        for i_pair, r in enumerate(reversed(song_similarity)):
            if r.relation < 1:
                break

            if len(self.ordering) == 0:
                self.ordering = [r.x, r.y]
                continue

            # Close as possible to its partner.
            if r.x not in self.ordering:
                i_end = r.y in self.ordering and self.ordering.index(r.y) or len(self.ordering)-1
                for i in range(i_end-1, -1, -1):
                    s = self.ordering[i]
                    # Fucked-up Mean Value Theorem?
                    r_comp = self.relationship_lookup(r.x, s)
                    if False:
                        print(f'\t{i:3d}: {r_comp}')
                    if (r_comp.relation is not None) and (r_comp.relation < Tournament.MONO_THRESHOLD):
                        break
                self.ordering.insert(i+1, r.x)
                if False and (i_pair < 50):
                    print(f'{r.pair_title()}: {i} (Lower) -> {[s for s in self.ordering]}')
            if r.y not in self.ordering:
                i_start = r.x in self.ordering and self.ordering.index(r.x) or 0
                for i in range(i_start, len(self.ordering)):
                    s = self.ordering[i]
                    # Fucked-up Mean Value Theorem?
                    r_comp = self.relationship_lookup(s, r.y)
                    if False:
                        print(f'\t{i:3d}: {r_comp}')
                    if (r_comp.relation is not None) and (r_comp.relation < Tournament.MONO_THRESHOLD):
                        break
                self.ordering.insert(i, r.y)
                if False and (i_pair < 50):
                    print(f'{r.pair_title()}: {i} (Upper) -> {[s for s in self.ordering]}')

        self.view_monotonicity(verbal, visual)

    def order_refine_monotonic(self, verbal=_VERBAL, visual=_VISUAL):
        # A few iterations of plain bubble sort until
        # the worst-ordered offenders are smoothed out.
        for a in range(Tournament.ITERATIONS_MONOTONIC_SORT):
            prev_ordering = [s for s in self.ordering]
            for i in range(len(self.ordering)-1):
                # Bubble sort!
                x = self.ordering[i]
                y = self.ordering[i+1]
                r_fwd = self.relationship_lookup(x, y, allow_link=True)
                r_rev = self.relationship_lookup(y, x, allow_link=True)
                if r_fwd and r_fwd.relation and r_fwd.relation < Tournament.MONO_THRESHOLD:
                    # Might benefit from a swap...
                    if r_rev and r_rev.relation and r_rev.relation > r_fwd.relation:
                        # Only swap if an improvement would be observed!
                        self.ordering = self.ordering[:i] + [y, x] + self.ordering[i+2:]
            if verbal:
                # List the changes in the ordering.
                for i, (prev, next) in enumerate(zip(prev_ordering, self.ordering)):
                    if prev.s_id != next.s_id:
                        print(f'#{i:3d}: {prev} -> {next}')

        self.view_monotonicity(verbal, visual)

    def order_refine_by_spice(self, verbal=_VERBAL, visual=_VISUAL):
        # Generate the naive spice rating for each chart.
        spice_list = []
        for i, song in enumerate(self.ordering):
            if i == 0:
                spice_list = [1]               # Minimum spice is 0.0
                continue

            i_nearest = i-1
            r = self.relationship_lookup(self.ordering[i_nearest], self.ordering[i])
            while (r.relation is None) and (i_nearest > 0):
                i_nearest -= 1
                r = self.relationship_lookup(self.ordering[i_nearest], self.ordering[i])

            if r.relation is None:
                # No helpful relationships - just pretend the spice is identical to the last one.
                spice_list.append(spice_list[-1])
            else:
                spice_list.append(spice_list[i_nearest] * r.relation)

        # Converge the spice rating by comparing with charts that are "close".
        # Re-sort the ordering according to the new spice ratings each time.
        convergence = [np.log2(spice_list[-1])]

        for k in range(Tournament.ITERATIONS_SCOBILITY_SORT):
            ordering_prev = [v for v in self.ordering]
            spice_prev = [v for v in spice_list]
            for i, song in enumerate(self.ordering):
                # Snip out the spice influence window.
                window    = self.ordering[max(0, i - Tournament.SCOBILITY_WINDOW_LOWER) : min(i + Tournament.SCOBILITY_WINDOW_UPPER + 1, len(self.ordering))]
                nearby_spice = spice_prev[max(0, i - Tournament.SCOBILITY_WINDOW_LOWER) : min(i + Tournament.SCOBILITY_WINDOW_UPPER + 1, len(self.ordering))]

                # Check relationships with the pivot chart.
                rel_window = [self.relationship_lookup(w, song) for w in window]

                # Calculate the influence of each nearby chart (including the pivot).
                nearby_relation = [r.relation for r in rel_window]
                nearby_strength = [r.strength for r in rel_window]
                nearby_contrib = [r * s * v for r, s, v in zip(nearby_relation, nearby_strength, nearby_spice) if s is not None]

                # Replace the spice with the window-averaged value.
                influence = sum(nearby_contrib) / sum([s for s in nearby_strength if s is not None])
                spice_list[i] = influence

            # Sort the chart list again based on the new spice values.
            spice_sort = [x for x in zip(spice_list, self.ordering)]
            spice_sort.sort(key=lambda x: x[0])
            spice_list = [x[0] / spice_sort[0][0] for x in spice_sort]          # Keep minimum spice pinned at 0.0
            self.ordering = [x[1] for x in spice_sort]

            convergence.append(np.log2(spice_list[-1]))
            if verbal:
                print(f'>>> Iteration {k+1:2d}: max spice = {convergence[-1]:0.6f}')
                if False:
                    for i, (prev, next) in enumerate(zip(ordering_last, self.ordering)):
                        if prev.s_id != next.s_id:
                            print(f'Iteration {k+1:2d}... #{i:3d}: {prev} -> {next}')

            for i, song in enumerate(self.ordering):
                song.spice = spice_list[i]

        return convergence

    def view_spice_ranking(self, fp=None):
        for s in self.ordering:
            print(f"{np.log2(s.spice):5.3f}🌶 {str(s):>60s}", file=fp or sys.stdout)

    def view_pvs_ranking(self, fp=None):
        pvs = sorted([s for s in self.ordering], key=lambda s: s.value / s.spice, reverse=True)
        for s in pvs:
            print(f"{s.value / s.spice:5.0f} pts / 2^🌶    ({s.value:4.0f} max points, {np.log2(s.spice):5.3f}🌶) {str(s):>80s}", file=fp or sys.stdout)

    def calc_point_curve(self, v: np.ndarray) -> np.ndarray:
        if self.POINT_CURVE == 'itl2023':
            log_base = 1.1032889141348
            pow_base = 61
            inflect = 50
        elif self.POINT_CURVE == 'itl2022':
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
    

    def calc_point_curve_inv(self, p: np.ndarray) -> np.ndarray:
        if self.POINT_CURVE == 'itl2023':
            log_base = 1.1032889141348
            pow_base = 61
            inflect = 50
        elif self.POINT_CURVE == 'itl2022':
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
        return v


    def calc_recommendations(self, player: Player, dst_dir=None, verbal=_VERBAL, visual=_VISUAL, use_player_name=False):
        if player.scobility is None or player.comfort_zone is None or player.timing_power is None:
            raise Exception(f'Can\'t recommend charts to {player.name} (#{player.e_id}) without first calculating scobility')

        scores = np.full((len(self.ordering),), np.nan)
        spices = np.full((len(self.ordering),), np.nan)
        values = np.full((len(self.ordering),), np.nan)

        for i, s in enumerate(self.ordering):
            if s.s_id in player.scores:
                scores[i] = player.scores[s.s_id].value
            if s.spice:
                spices[i] = s.spice
                values[i] = s.value

        # The lowest possible score (0%) on the chart with
        # the lowest spice level will define the (0, 0) point.
        p_min = np.log2(Tournament.PERFECT_OFFSET + 1)
        p_max = np.log2(Tournament.PERFECT_OFFSET)
        p_spices = np.log2(spices)
        p_scores = np.log2(Tournament.PERFECT_OFFSET + scores) - p_min
        
        # Determine the "top N" cutoff.
        points_all = [self.songs[s].value * self.calc_point_curve(100 - 100*v.value) * 0.01 for s, v in player.scores.items()]
        points_all.sort(reverse=True)
        if len(points_all) < Tournament.RANKING_CHART_COUNT:
            tourney_rp_cutoff = 0
        else:
            tourney_rp_cutoff = points_all[Tournament.RANKING_CHART_COUNT]

        points = np.round(values * self.calc_point_curve(100 - 100*scores) * 0.01)          # Current tournament points (not ranking points!) from this song
        pred_qual = player.timing_power + p_spices * player.comfort_zone                    # Score quality that would bring this chart up to the player's scobility fit
        # Missing %EX score that would bring this chart up to the player's scobility fit
        ex_target = np.clip((np.power(2, p_spices - pred_qual)) * (Tournament.PERFECT_OFFSET + 1), a_min=0, a_max=1)
        pt_access = np.round(values * self.calc_point_curve(100 - 100*ex_target) * 0.01) - points  # Additional point gain (not tournament RP!) from bringing this chart up
        with np.errstate(divide='ignore', invalid='ignore'):
            ex_rp_hit = 1 - 0.01*self.calc_point_curve_inv(100 * tourney_rp_cutoff / values)
            need_qual = p_spices - np.log2(Tournament.PERFECT_OFFSET + ex_rp_hit) + p_min

        # Draw a best-fit line to calculate the player's strength.
        # Mostly used to determine the "comfort zone", i.e.
        # where do you outperform your peers? easier or harder charts?
        # by observing the slope of the best-fit line.
        p_played = ~np.isnan(p_scores)

        if sum(p_played) < 2:       # TODO: parameterize this limit
            raise Exception(f'{player.name} (#{player.e_id}) hasn\'t played enough charts with a known spice rating yet')
                
        stats = StringIO()
        print(f'Scobility {_VERSION} for {player}: {player.scobility:0.3f}🌶', file=stats)
        if player.comfort_zone > 0:
            print(f'>>> You outperform your peers on harder charts. (m = {player.comfort_zone:0.3f})', file=stats)
        else:
            print(f'>>> You outperform your peers on easier charts. (m = {player.comfort_zone:0.3f})', file=stats)

        # Give various recommendations to the player on which songs to redo,
        # as well as the target score that brings up the scobility quality
        # to the player's skill trend.
        p_quality = p_spices[p_played] - p_scores[p_played]
        p_songs = np.array(self.ordering)[p_played]
        p_ranking = [z for z in zip(
            [s for s in p_songs],
            [q for q in p_quality],
            [p for p in points[p_played]],
            [x for x in ex_target[p_played]],
            [p for p in pt_access[p_played]],
            [v for v in scores[p_played]],
            [x for x in ex_rp_hit[p_played]],
            [q for q in need_qual[p_played]],
            [q for q in pred_qual[p_played]],
        )]

        # Plain ol' "these are your worst/best quality songs so far"
        p_ranking.sort(key=lambda x: x[1])
        print(f'\nTop {Tournament.MAX_HEADTAIL_QUALITY} Best Scores for {player}:', file=stats)
        for z in reversed(p_ranking[-Tournament.MAX_HEADTAIL_QUALITY:]):
            print(f'{(1 - z[5])*100:5.2f}% on {z[0].name}, worth {z[2]:.0f} / {z[0].value:.0f} points (Quality parameter: {z[1]:0.3f})', file=stats)
        print(f'\nTop {Tournament.MAX_HEADTAIL_QUALITY} Improvement Opportunities for {player}:', file=stats)
        for z in p_ranking[:Tournament.MAX_HEADTAIL_QUALITY]:
            print(f'{(1 - z[5])*100:5.2f}% on {z[0].name}, worth {z[2]:.0f} / {z[0].value:.0f} points (Quality parameter: {z[1]:0.3f})', file=stats)
            print(f'>>> Based on your scobility, aim for a score of {(1 - z[3])*100:5.2f}% on {z[0].name}, worth {z[2]+z[4]:.0f} points.', file=stats)

        # What could raise your tourney RP?
        # First, determine the "top N" cutoff.
        # Then calculate how much "real" RP gain is achievable.
        # If the chart wasn't already in the top 75,
        # catch-up points should be deducted before ranking its impact.
        p_valuable = [z for z in p_ranking if (z[2] + z[4] >= tourney_rp_cutoff) and (z[4] > 0)]
        p_valuable.sort(key=lambda z: z[4] + min(z[2]-tourney_rp_cutoff, 0), reverse=True)
        n_rp_rec = 1000 #int(np.power(len(points_all), 2/3))
        # If there aren't enough charts with catch-up points,
        # start recommending charts that are just above the scobility trend,
        # as well as what EX score would be required to gain RP.
        p_almost_valuable = [z for z in p_ranking if
            (z[4] < 0) and                  # The current score is already at or above scobility trend
            (z[2] <= tourney_rp_cutoff) and # The points currently held are under the RP cutoff 
            (z[6] > 0) and                  # The score is possible (i.e., <= 100% EX)
            (z[6] < z[5])                   # The score required to reach RP would be an improvement
        ]
        # p_songs_worth = [(
        #     s,
        #     100 * 100 - v.value,
        #     self.calc_point_curve(100 * 100 - v.value),
        #     self.songs[s].value,
        #     self.calc_point_curve_inv(100 * tourney_rp_cutoff / self.songs[s].value)
        # ) for s, v in player.scores.items() if self.songs[s].value > 0]
        # p_almost_valuable = [z for z in p_songs_worth if 
        #     (z[4] > z[1]) and           # The score required to reach RP would be an improvement
        #     (z[4] <= 100) and           # The score is possible (i.e., <= 100% EX)
        #     (z[3] < tourney_rp_cutoff)  # The points currently held are under the RP cutoff
        # ]
        p_almost_valuable.sort(key=lambda z: z[7]-z[8])
        n_recs_available = min(n_rp_rec, len(p_valuable) + len(p_almost_valuable))

        print(f'\nTop {n_recs_available} Tourney RP Improvement Opportunities for {player}:', file=stats)
        if tourney_rp_cutoff == 0:
            print(f'\tYour top {Tournament.RANKING_CHART_COUNT} still has room for completely new additions!', file=stats)
            print('\tBut, if you want scobility recommendations on what to replay, read on...', file=stats)
        else:
            print(f'\tYour current top {Tournament.RANKING_CHART_COUNT} cutoff is {tourney_rp_cutoff:.0f} points.', file=stats)
        print(f'\tTarget scores are calculated to match your personal current scobility.', file=stats)
        print(f'\tAchieve the target score on any of the listed charts and gain RP!', file=stats)
        for z in p_valuable[:n_rp_rec]:
            print(f'{z[4] + min(z[2]-tourney_rp_cutoff, 0):+5.0f} RP: Raise {z[0].name} from {(1 - z[5])*100:5.2f}% ({z[2]:.0f}pt.) to at least {(1 - z[3])*100:5.2f}% ({z[2]+z[4]:.0f}pt.)', file=stats)
        if len(p_valuable) < n_rp_rec:
            print(f'\tThat\'s all the raises scobility can directly identify for now.', file=stats)
            print(f'\tYour scores on the following charts already meet your personal current scobility,', file=stats)
            print(f'\tbut if you can raise them a bit more, you could gain RP from these too.', file=stats)
            print(f'\tAnd don\'t forget to try charts you haven\'t played yet!', file=stats)
            for z in p_almost_valuable[:n_rp_rec-len(p_valuable)]:
                print(f'Raise {z[0].name} from {(1 - z[5])*100:5.2f}% ({z[2]:.0f}pt.) to at least {(1 - z[6])*100:5.2f}% (New quality parameter: {z[7]:0.3f})', file=stats)
        

        if dst_dir is not None:
            if use_player_name:
                dst_log = os.path.join(dst_dir, f'{player.e_id}-{slugify(player.name)}.txt')
            else:
                dst_log = os.path.join(dst_dir, f'{player.e_id}.txt')
            with open(dst_log, 'w', encoding='utf-8') as fp:
                fp.write(stats.getvalue())
        if verbal:
            print(stats.getvalue())

    def calc_player_scobility(self, player: Player, dst_dir=None, verbal=_VERBAL, visual=_VISUAL, use_player_name=False):
        scores = np.full((len(self.ordering),), np.nan)
        spices = np.full((len(self.ordering),), np.nan)
        counts = np.full((len(self.ordering),), np.nan)
        dates  = np.full((len(self.ordering),), np.nan)

        for i, s in enumerate(self.ordering):
            if s.s_id in player.scores:
                scores[i] = player.scores[s.s_id].value
                counts[i] = player.scores[s.s_id].plays
                try:
                    dates[i] = (player.scores[s.s_id].last_played - dt.utcfromtimestamp(0)).total_seconds()
                except:
                    dates[i] = 0
            if s.spice:
                spices[i] = s.spice

        # The lowest possible score (0%) on the chart with
        # the lowest spice level will define the (0, 0) point.
        p_min = np.log2(Tournament.PERFECT_OFFSET + 1)
        p_max = np.log2(Tournament.PERFECT_OFFSET)
        p_spices = np.log2(spices)
        p_scores = np.log2(Tournament.PERFECT_OFFSET + scores) - p_min

        # Draw a best-fit line to calculate the player's strength.
        # Mostly used to determine the "comfort zone", i.e.
        # where do you outperform your peers? easier or harder charts?
        # by observing the slope of the best-fit line.
        p_played = ~np.isnan(p_scores)

        if sum(p_played) < 5:       # TODO: parameterize this limit
            raise Exception(f'{player.name} (#{player.e_id}) hasn\'t played enough charts with a known spice rating yet')

        a = p_spices[p_played]
        b = p_spices[p_played] - p_scores[p_played]         # Expected to be constant...
        w = a # np.ones_like(a) # 1 - 1/(a + Relationship.WEIGHT_OFFSET)**2
        m = np.vstack([np.ones_like(a), a])
        coefs = np.linalg.lstsq(m.T * w[:, np.newaxis], b * w, rcond=None)[0]
        b_eq = coefs @ m

        p_quality = p_spices[p_played] - p_scores[p_played]
        p_counts = counts[p_played]
        p_dates = dates[p_played]

        # TODO: derive a volforce-like "tournament power" that rewards
        # playing more songs as well as getting better scores
        # this is a pretty silly first stab at it imo
        player.tourney_power = sum(p_quality) / len(p_quality) * np.sqrt(len(p_quality) / len(self.ordering)) * Tournament.MAX_TOURNEY_POWER
        player.scobility = sum(p_quality) / len(p_quality)  # Simple average...
        player.timing_power = coefs[0]
        player.comfort_zone = coefs[1]

        # Plot score quality by chart spice level
        pts_ordered = sorted(zip(
            p_dates,
            a,
            b,
            np.power(np.clip(p_counts-0.6, a_min=0, a_max=None), 0.6)*48,
            Tournament.colorize_dates(p_dates)
        ), key=lambda x: x[0])
        # Ensure the points are plotted from oldest to newest
        # TODO: point size scaling and specialized color gradient
        plt.subplots(figsize=(6, 6))
        plt.scatter(
            [p[1] for p in pts_ordered],
            [p[2] for p in pts_ordered],
            s=[p[3] for p in pts_ordered],
            c=[p[4] for p in pts_ordered]
        )
        plt.plot(a, b_eq, color='tab:blue')
        plt.xlabel('Chart spice level')
        plt.ylabel('Score quality')
        plt.title(f'Scobility {_VERSION} plot for {player}\nRating: $\\bf{{{player.scobility:0.3f}}}$')
        if dst_dir is not None:
            if use_player_name:
                plt.savefig(os.path.join(dst_dir, f'{player.e_id}-{slugify(player.name)}.png'))
            else:
                plt.savefig(os.path.join(dst_dir, f'{player.e_id}.png'))
        if visual:
            plt.show()
        plt.close('all')


    def view_scobility_ranking(self, fp=None):
        p_sort = sorted([p for p in self.players.values() if p.scobility is not None], key=lambda p: p.scobility)
        for p in reversed(p_sort):
            print(f"{p.scobility:6.3f}🌶 {str(p):>30s} (balance: {p.comfort_zone:6.3f})", file=fp or sys.stdout)

    def view_tourney_ranking(self, fp=None):
        p_sort = sorted([p for p in self.players.values() if p.tourney_power is not None], key=lambda p: p.tourney_power)
        for p in reversed(p_sort):
            print(f"{p.tourney_power:7.3f} TP {str(p):>30s} (scobility: {p.scobility:6.3f}🌶, balance: {p.comfort_zone:6.3f})", file=fp or sys.stdout)


def load_json_data(root='itl_data', jit=False):
    tourney = Tournament()

    if not os.path.exists(root):
        raise FileNotFoundError(f'Tournament data scrape needs to be unpacked to {root}!')

    # Song info
    song_files = glob.glob(os.path.join(root, 'song_info', '*.json'))
    for fn in song_files:
        with open(fn, 'r') as fp:
            song_data = json.load(fp)
            s = Song(song_data['song'])
            tourney.songs[s.s_id] = s
            del song_data

    # Player info
    player_files = glob.glob(os.path.join(root, 'entrant_info', '*.json'))
    for fn in player_files:
        with open(fn, 'r') as fp:
            player_data = json.load(fp)
            p = Player(player_data['entrant'])
            tourney.players[p.e_id] = p
            del player_data

    # Score data
    if not jit:
        score_files = glob.glob(os.path.join(root, 'song_scores', '*.json'))
        for fn in score_files:
            with open(fn, 'r') as fp:
                score_data = json.load(fp)
                tourney.load_score_data(score_data['scores'])
                del score_data

    return tourney


def process(src='itl2024', force_recalculate_spice: bool = False):
    src = src.lower()
    scrape_designator = ''
    tourney_fn = None

    print('========================================================================')
    print(f'=== Scobility {_VERSION}')
    print('========================================================================')
    print()
    if src == 'itl2022':
        # https://github.com/G-Wen/itl_history
        jit = False
        root = 'itl_data'
    elif src == 'itl2023':
        # Personally scraped
        jit = False
        latest_itl2023 = sorted([d for d in os.listdir('itl2023_data') if re.match('^\d+$', d)])[-1]
        scrape_designator = '_' + latest_itl2023
        root = os.path.join('itl2023_data', latest_itl2023)
    elif src == 'itl2024':
        # Personally scraped
        jit = False
        latest_itl2024 = sorted([d for d in os.listdir('itl2024_prep') if re.match('^\d+$', d)])[-1]
        scrape_designator = '_' + latest_itl2024
        root = os.path.join('itl2024_prep', latest_itl2024)
    elif src == '3ic':
        # Privately provided
        jit = True
        root = '3ic_data'

    tourney_list = sorted([d for d in os.listdir('.') if re.match(f'^scobility_{src}.*\.json$', d)])
    tourney = Tournament()
    if len(tourney_list) == 0 or force_recalculate_spice:
        print('========================================================================')
        print('=== Loading data...')
        tourney = load_json_data(root=root, jit=jit)
        print('========================================================================')
        print('=== Setting up relationships...')
        tourney.setup_relationships()
        print('========================================================================')
        print('=== Calculating relationships...')
        if jit:
            tourney.calc_relationships_jit(src='3ic_data/song_scores', verbal=False)
        else:
            tourney.calc_relationships(verbal=False)
        print('========================================================================')
        print('=== Setting up closest-neighbor initial order...')
        tourney.order_songs_initial(verbal=False, visual=False)
        print('========================================================================')
        print('=== Bubbling out non-monotonicities...')
        tourney.order_refine_monotonic(verbal=False, visual=False)
        print('========================================================================')
        print('=== Refining spice ranking using neighborhood influence...')
        tourney.order_refine_by_spice(verbal=True, visual=False)
        print('========================================================================')
        print('=== Spice ranking calculation complete!...')
        # itl.view_spice_ranking()
        with open(f'{src}_data/spice_ranking{scrape_designator}.txt', 'w', encoding='utf-8') as fp:
            tourney.view_spice_ranking(fp)

        # Store (and re-load?)
        tourney_fn = f'scobility_{src}{scrape_designator}.json'
        with open(tourney_fn, 'w') as fp:
            json.dump(tourney.dump(), fp, indent='\t')
        print('========================================================================')
        print(f'=== Saved tournament snapshot to {tourney_fn}')
    else:
        # Reload
        tourney_fn = tourney_list[-1]
        print('========================================================================')
        print(f'=== Loading tournament snapshot from {tourney_fn}...')
        with open(tourney_fn, 'r') as fp:
            tourney = Tournament.load(json.load(fp))

    with open(f'{src}_data/pvs_ranking{scrape_designator}.txt', 'w', encoding='utf-8') as fp:
        tourney.view_pvs_ranking(fp)
    print('========================================================================')
    print('=== Performing scobility calculations...')
    output_dir = os.path.join(f'{src}_data', scrape_designator[1:], 'scobility')
    os.makedirs(output_dir, exist_ok=True)
    for p in tourney.players.values():
        try:
            tourney.calc_player_scobility(p, dst_dir=output_dir, verbal=False, visual=False, use_player_name=True)
            tourney.calc_recommendations(p,  dst_dir=output_dir, verbal=False, visual=False, use_player_name=True)
        except Exception as e:
            print(f'Scobility calculation failed for {p} (probably due to lack of sufficient score data)', file=sys.stderr)
            tb.print_exc(file=sys.stderr)
    print('========================================================================')
    print('=== Ranking players by scobility...')
    # itl.view_scobility_ranking()
    with open(os.path.join(f'{src}_data', f'scobility_ranking{scrape_designator}.txt'), 'w', encoding='utf-8') as fp:
        tourney.view_scobility_ranking(fp)
    print('========================================================================')
    print('=== Ranking players by tourney performance...')
    # itl.view_tourney_ranking()
    with open(os.path.join(f'{src}_data', f'tourney_ranking{scrape_designator}.txt'), 'w', encoding='utf-8') as fp:
        tourney.view_tourney_ranking(fp)
    print('========================================================================')
    print('=== Done!')
    
    # Store (and re-load?)
    tourney_fn = f'scobility_{src}{scrape_designator}.json'
    with open(tourney_fn, 'w') as fp:
        json.dump(tourney.dump(), fp, indent='\t')

    return tourney


if __name__ == '__main__':
    process(src='itl2024', force_recalculate_spice=True)
    