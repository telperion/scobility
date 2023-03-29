"""Scobility! :chili_pepper:"""

import json
import glob
import os
import re
import sys
import gc
from io import StringIO
from functools import reduce
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List

_VERSION = 'v0.97'
_VERBAL = False
_VISUAL = False

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
    clear: Clear = Clear.PASS
    value: float = 0

    def dump(self):
        return {
            's_id': self.s_id,
            'e_id': self.e_id,
            'clear': int(self.clear),
            'value': self.value
        }
    
    @classmethod
    def load(cls, data):
        cls.s_id = data['s_id']
        cls.e_id = data['e_id']
        cls.clear = Clear(data['clear'])
        cls.value = data['value']

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
    scores: dict = field(default_factory=dict)      # e_id: Score
    spice: float = None                             # Not a Dune reference. capsaicin not cinnamon

    def __init__(self, data: dict):
        try:
            self.s_id = data['song_id']
            self.hash = data['song_hash']

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
            'spice': self.spice
        }

    @classmethod
    def load(cls, data):
        cls.s_id = data['s_id']
        cls.hash = data['hash']
        cls.title = data['title']
        cls.subtitle = data['subtitle']
        cls.artist = data['artist']
        cls.meter = data['meter']
        cls.slot = data['slot']
        cls.spice = data['spice']

    def __str__(self):
        return f"#{self.s_id} {self.title} ({self.slot} {self.meter})"


@dataclass
class Player:
    name: str
    e_id: int = -1
    g_id: int = -1
    scores: dict = field(default_factory=dict)      # s_id: Score

    scobility: float = None
    comfort_zone: float = None
    tourney_power: float = None

    def __init__(self, data: dict):
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
        self.tourney_power = None

    def dump(self):
        return {
            'name': self.name,
            'e_id': self.e_id,
            'g_id': self.g_id,
            'scobility': self.scobility,
            'comfort_zone': self.comfort_zone,
            'tourney_power': self.tourney_power
        }

    @classmethod
    def load(cls, data):
        cls.name = data['name']
        cls.e_id = data['e_id']
        cls.g_id = data['g_id']
        cls.scobility = data['scobility']
        cls.comfort_zone = data['comfort_zone']
        cls.tourney_power = data['tourney_power']

    def __str__(self):
        return f"{self.name} (#{self.e_id})"


class Relationship:
    MAX_NEG_LIMIT = 0.3             # i.e., 700,000 min EX score
    MIN_NEG_LIMIT = 0.0002          # i.e., 999,800 max EX score (ONLY for initial score scaling!)
    WEIGHT_OFFSET = 0.5
    MIN_COMMON_PLAYERS = 10
    
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
        cls.x = songs[data['x_id']]
        cls.y = songs[data['y_id']]
        x_players = set(cls.x.scores)
        y_players = set(cls.y.scores)
        cls.e_common = x_players.intersection(y_players)
        cls.relation = data['relation']
        cls.strength = data['strength']


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
    MIN_COMMON_PLAYERS = 50                         # For ordering purposes
    ITERATIONS_MONOTONIC_SORT = 10                  # Bubble sort for correlation factor monotonicity
    ITERATIONS_SCOBILITY_SORT = 500                 # Refining the scobility values and post-sorting
    SCOBILITY_WINDOW_LOWER = 10                     # Incorporate this many lower scobility readings
    SCOBILITY_WINDOW_UPPER = 6                      # Incorporate this many higher scobility readings
    PERFECT_OFFSET = 0.003                          # Push scores away from logarithmic asymptote
    MAX_TOURNEY_POWER = 100                         # idk, I need a value

    players: dict = field(default_factory=dict)         # e_id: Player
    songs: dict = field(default_factory=dict)           # s_id: Song
    relationships: dict = field(default_factory=dict)   # r.x: {r.y: Relationship}
    ordering: list = field(default_factory=list)        # List[Song]

    def dump(self):
        j = {
            'songs': [],
            'players': [],
            'scores': [],
            'relationships': []
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

        return j

    @classmethod
    def load(cls, data):
        for s_data in data['songs']:
            s = Song.load(s_data)
            cls.songs[s.s_id] = s
        for p_data in data['players']:
            p = Player.load(p_data)
            cls.players[p.e_id] = p
        for v_data in data['scores']:
            v = Song.load(v_data)
            if v.s_id in cls.songs:
                cls.songs[v.s_id].scores[v.e_id] = v
            if v.e_id in cls.players:
                cls.players[v.e_id].scores[v.s_id] = v
        for r_data in data['relationships']:
            r = Relationship.load(r_data, cls.songs)
            if r.x.s_id not in cls.relationships:
                cls.relationships[r.x.s_id] = {}
            cls.relationships[r.x.s_id][r.y.s_id] = r

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


    def load_score_data(self, score_data: list, assign_to_player: bool = True):
        for s_data in score_data:
            try:
                s = Score(
                    s_id = s_data['song_id'],
                    e_id = s_data['entrant_id'],
                    clear = Clear(s_data['score_best_clear_type']),
                    value = 1 - s_data['score_ex'] * Score.SCORE_SCALAR
                )
            except:
                s = Score(
                    s_id = s_data['chartId'],
                    e_id = s_data['entrantId'],
                    clear = Clear(s_data['clearType']),
                    value = 1 - s_data['ex'] * Score.SCORE_SCALAR
                )

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
            r = self.relationship_lookup(self.ordering[i-1], self.ordering[i])
            spice_list.append(spice_list[-1] * r.relation)

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


    def calc_player_scobility(self, player: Player, dst_dir=None, verbal=_VERBAL, visual=_VISUAL, use_player_name=False):
        scores = np.full((len(self.ordering),), np.nan)
        spices = np.full((len(self.ordering),), np.nan)

        for i, s in enumerate(self.ordering):
            if s.s_id in player.scores:
                scores[i] = player.scores[s.s_id].value
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
        a = p_spices[p_played]
        b = p_spices[p_played] - p_scores[p_played]         # Expected to be constant...
        w = a # np.ones_like(a) # 1 - 1/(a + Relationship.WEIGHT_OFFSET)**2
        m = np.vstack([np.ones_like(a), a])
        coefs = np.linalg.lstsq(m.T * w[:, np.newaxis], b * w, rcond=None)[0]
        b_eq = coefs @ m

        p_quality = p_spices[p_played] - p_scores[p_played]
        p_songs = np.array(self.ordering)[p_played]
        p_ranking = [z for z in zip([s for s in p_songs], [q for q in p_quality])]
        p_ranking.sort(key=lambda x: x[1])

        # TODO: derive a volforce-like "tournament power" that rewards
        # playing more songs as well as getting better scores
        # this is a pretty silly first stab at it imo
        player.tourney_power = sum(p_quality) / len(p_quality) * np.sqrt(len(p_quality) / len(self.ordering)) * Tournament.MAX_TOURNEY_POWER
        player.scobility = sum(p_quality) / len(p_quality)  # Simple average...
        player.comfort_zone = coefs[1]

        # Plot score quality by chart spice level
        plt.subplots(figsize=(6, 6))
        plt.scatter(a, b, color='tab:pink')
        plt.plot(a, b_eq, color='tab:blue')
        plt.xlabel('Chart spice level')
        plt.ylabel('Score quality')
        plt.title(f'Scobility {_VERSION} plot for {player}\nRating: $\\bf{{{player.scobility:0.3f}}}$')
        if dst_dir is not None:
            if use_player_name:
                plt.savefig(os.path.join(dst_dir, f'{player.e_id}-{player.name}.png'))
            else:
                plt.savefig(os.path.join(dst_dir, f'{player.e_id}.png'))
        if visual:
            plt.show()
        plt.close('all')

        stats = StringIO()
        print(f'Scobility {_VERSION} for {player}: {player.scobility:0.3f}🌶', file=stats)
        if player.comfort_zone > 0:
            print(f'>>> You outperform your peers on harder charts. (m = {player.comfort_zone:0.3f})', file=stats)
        else:
            print(f'>>> You outperform your peers on easier charts. (m = {player.comfort_zone:0.3f})', file=stats)

        # TODO: back out what an "appropriate" score for this player
        # on the listed songs would be
        print(f'\nTop 5 Best Scores for {player}:', file=stats)
        for s, v in reversed(p_ranking[-5:]):
            print(f'{(1 - player.scores[s.s_id].value)*100:5.2f}% on {s} (Quality parameter: {v:0.3f})', file=stats)
        print(f'\nTop 5 Improvement Opportunities for {player}:', file=stats)
        for s, v in p_ranking[:5]:
            print(f'{(1 - player.scores[s.s_id].value)*100:5.2f}% on {s} (Quality parameter: {v:0.3f})', file=stats)

        if dst_dir is not None:
            if use_player_name:
                dst_log = os.path.join(dst_dir, f'{player.e_id}-{player.name}.txt')
            else:
                dst_log = os.path.join(dst_dir, f'{player.e_id}.txt')
            with open(dst_log, 'w', encoding='utf-8') as fp:
                fp.write(stats.getvalue())
        if verbal:
            print(stats.getvalue())


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


def process(src='itl2023'):
    print('========================================================================')
    print(f'=== Scobility {_VERSION}')
    print('========================================================================')
    print()
    print('========================================================================')
    print('=== Loading data...')
    if src.lower() == 'itl2022':
        # https://github.com/G-Wen/itl_history
        jit = False
        tourney = load_json_data(root='itl_data', jit=jit)
    elif src.lower() == 'itl2023':
        # Personally scraped
        jit = False
        latest_itl2023 = sorted([d for d in os.listdir('itl2023_data') if re.match('^\d+$', d)])[-1]
        tourney = load_json_data(root=os.path.join('itl2023_data', latest_itl2023), jit=jit)
    elif src.lower() == '3ic':
        # Privately provided
        jit = True
        tourney = load_json_data(root='3ic_data', jit=jit)
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
    with open(f'{src}_data/spice_ranking.txt', 'w', encoding='utf-8') as fp:
        tourney.view_spice_ranking(fp)

    # Store (and re-load?)
    # with open('scobility_itl.json', 'w') as fp:
    #     json.dump(itl.dump(), fp, indent='\t')
    
    # with open('scobility_itl.json', 'r') as fp:
    #     itl2 = Tournament.load(json.load(fp))

    print('========================================================================')
    print('=== Performing scobility calculations...')
    for p in tourney.players.values():
        try:
            tourney.calc_player_scobility(p, dst_dir=f'{src}_data/scobility', verbal=False, visual=False, use_player_name=True)
        except Exception as e:
            print(f'Scobility calculation failed for {p} (probably due to lack of sufficient score data)', file=sys.stderr)
            print(e, file=sys.stderr)
    print('========================================================================')
    print('=== Ranking players by scobility...')
    # itl.view_scobility_ranking()
    with open(f'{src}_data/scobility_ranking.txt', 'w', encoding='utf-8') as fp:
        tourney.view_scobility_ranking(fp)
    print('========================================================================')
    print('=== Ranking players by tourney performance...')
    # itl.view_tourney_ranking()
    with open(f'{src}_data/tourney_ranking.txt', 'w', encoding='utf-8') as fp:
        tourney.view_tourney_ranking(fp)
    print('========================================================================')
    print('=== Done!')

    return tourney


if __name__ == '__main__':
    process(src='itl2023')
    