"""Scobility! :chili_pepper:"""

import json
import glob
import os
from functools import reduce
from xml.dom import NotFoundErr

import numpy as np
from matplotlib import pyplot as plt

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

class Clear(IntEnum):
    PASS = 0
    FC = 1
    FEC = 2
    QUAD = 3
    QUINT = 4

@dataclass
class Score:
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

    def __init__(self, data: dict):
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

        self.scores = {}

    def dump(self):
        return {
            's_id': self.s_id,
            'hash': self.hash,
            'title': self.title,
            'subtitle': self.subtitle,
            'artist': self.artist,
            'meter': self.meter,
            'slot': self.slot
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

    def __str__(self):
        return f"#{self.s_id} {self.title} ({self.slot} {self.meter})"


@dataclass
class Player:
    name: str
    e_id: int = -1
    g_id: int = -1
    scores: dict = field(default_factory=dict)      # s_id: Score

    def __init__(self, data: dict):
        self.name = data['members_name']
        self.e_id = data['entrant_id']
        self.g_id = data['entrant_members_id']
        self.scores = {}

    def dump(self):
        return {
            'name': self.name,
            'e_id': self.e_id,
            'g_id': self.g_id
        }

    @classmethod
    def load(cls, data):
        cls.name = data['name']
        cls.e_id = data['e_id']
        cls.g_id = data['g_id']

    def __str__(self):
        return f'{self.name} (#{self.e_id})'


class Relationship:
    SCORE_SCALAR = 0.0001
    MAX_NEG_LIMIT = 0.20        # i.e., 80% min EX score
    MIN_NEG_LIMIT = 0.01        # i.e., 99% max EX score (ONLY for initial score scaling!)
    WEIGHT_OFFSET = 0.5
    MIN_COMMON_PLAYERS = 10
    VERBAL = False
    VISUAL = False
    
    def __init__(self, x: Song, y: Song):
        self.x = x
        self.y = y
        x_players = set(x.scores)
        y_players = set(y.scores)
        self.e_common = x_players.intersection(y_players)

        # TODO: Postpone calculation?
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
        if len(self.e_common) < Relationship.MIN_COMMON_PLAYERS:
            raise ValueError(f'Not enough players to relate {self.compare_title()} ({len(self.e_common)} players, need {Relationship.MIN_COMMON_PLAYERS})')

        x_scores = [self.x.scores[e_id].value for e_id in self.e_common]
        y_scores = [self.y.scores[e_id].value for e_id in self.e_common]

        ex_matrix = 1 - np.vstack([x_scores, y_scores]) * Relationship.SCORE_SCALAR

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

        # plotting is funny
        z = np.sort(np.vstack((x_col, y_col)), axis=-1)
        
        # slope of line thru 0 as a starting point?
        a = x_col
        b = y_col
        w = 1 - 1/(a + Relationship.WEIGHT_OFFSET)**2
        p = np.vstack([a])
        coefs, resid = np.linalg.lstsq(p.T * w[:, np.newaxis], b * w, rcond=None)[:2]
        b_eq = coefs @ p

        self.relation = coefs[0]                        # Slope of the best-fit line
        self.strength = np.sqrt(len(a) / resid[0])      # Reciprocal of residual, accounting for number of data points
        
        if Relationship.VERBAL:
            print(self)

        if Relationship.VISUAL:
            plt.scatter(a, b_eq)

            x_all = ex_matrix_screen[0, :]
            y_all = ex_matrix_screen[1, :]

            plt.scatter(x_all, y_all)
            plt.show()
        


@dataclass
class Tournament:
    MONO_THRESHOLD = 0.999999                       # Monotonicity check
    MIN_COMMON_PLAYERS = 50                         # For ordering purposes
    ITERATIONS_MONOTONIC_SORT = 10                  # Bubble sort for correlation factor monotonicity
    ITERATIONS_SCOBILITY_SORT = 500                 # Refining the scobility values and post-sorting
    SCOBILITY_WINDOW_LOWER = 10                     # Incorporate this many lower scobility readings
    SCOBILITY_WINDOW_UPPER = 6                      # Incorporate this many higher scobility readings

    players: dict = field(default_factory=dict)         # e_id: Player
    songs: dict = field(default_factory=dict)           # s_id: Song
    relationships: dict = field(default_factory=dict)   # r.x: {r.y: Relationship}
    ordering: list = field(default_factory=list)        # List[Song]
    scobility: list = field(default_factory=list)       # List[float]

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
        if len(links) == 0:
            return None
        if len(links) == 1:
            return links[0]
        
        return reduce(Tournament.link_through, links)


    def load_score_data(self, score_data: list):
        for s_data in score_data:
            s = Score(
                s_id = s_data['song_id'],
                e_id = s_data['entrant_id'],
                clear = Clear(s_data['score_best_clear_type']),
                value = s_data['score_ex']
            )
            if s.s_id in self.songs:
                self.songs[s.s_id].scores[s.e_id] = s
            if s.e_id in self.players:
                self.players[s.e_id].scores[s.s_id] = s

    def setup_relationships(self):
        for x in self.songs:
            for y in self.songs:
                if x != y:
                    if x not in self.relationships:
                        self.relationships[x] = {}
                    self.relationships[x][y] = Relationship(self.songs[x], self.songs[y])

    def calc_relationships(self, verbal=Relationship.VERBAL):
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

    def relationship_lookup(self, x: Song, y: Song, allow_link: bool = False) -> Relationship:
        # TODO: Oh no
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
            potential_links = []
            for z_id in self.relationships[x.s_id]:
                if y.s_id in self.relationships[z_id]:
                    potential_links.append(Tournament.link_through(self.relationships[x.s_id][z_id], self.relationships[z_id][y.s_id]))
            if len(potential_links) < 1:
                raise ValueError(f'No relationship for {y} vs. {x}, even with linking!')
            potential_links.sort(key=lambda r: r.strength)
            return potential_links[-1]

        return self.relationships[x.s_id][y.s_id]

    def view_monotonicity(self, verbal=Relationship.VERBAL, visual=Relationship.VISUAL):
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

    def order_songs_initial(self, verbal=Relationship.VERBAL, visual=Relationship.VISUAL):
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

    def order_refine_monotonic(self, verbal=Relationship.VERBAL, visual=Relationship.VISUAL):
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
                if r_fwd and r_fwd.relation < Tournament.MONO_THRESHOLD:
                    # Might benefit from a swap...
                    if r_rev and r_rev.relation > r_fwd.relation:
                        # Only swap if an improvement would be observed!
                        self.ordering = self.ordering[:i] + [y, x] + self.ordering[i+2:]
            if verbal:
                # List the changes in the ordering.
                for i, (prev, next) in enumerate(zip(prev_ordering, self.ordering)):
                    if prev.s_id != next.s_id:
                        print(f'#{i:3d}: {prev} -> {next}')

        self.view_monotonicity(verbal, visual)

    def order_refine_scobility(self, verbal=Relationship.VERBAL, visual=Relationship.VISUAL):
        # Generate the naive scobility rating for each chart.
        self.scobility = []
        for i, s in enumerate(self.ordering):
            if i == 0:
                self.scobility = [1]               # Minimum scobility is 0.0
                continue
            r = self.relationship_lookup(self.ordering[i-1], self.ordering[i])
            self.scobility.append(self.scobility[-1] * r.relation)

        # Converge the scobility rating by comparing with charts that are "close".
        # Re-sort the ordering according to the new scobility ratings each time.
        convergence = [np.log2(self.scobility[-1])]

        for k in range(Tournament.ITERATIONS_SCOBILITY_SORT):
            ordering_last = [v for v in self.ordering]
            scobility_last = [v for v in self.scobility]
            for i, s in enumerate(self.ordering):
                # Snip out the scobility influence window.
                window = self.ordering[max(0, i - Tournament.SCOBILITY_WINDOW_LOWER) : min(i + Tournament.SCOBILITY_WINDOW_UPPER + 1, len(self.ordering))]
                nearby_scobility = scobility_last[max(0, i - Tournament.SCOBILITY_WINDOW_LOWER) : min(i + Tournament.SCOBILITY_WINDOW_UPPER + 1, len(self.ordering))]

                # Check relationships with the pivot chart.
                rel_window = [self.relationship_lookup(w, s) for w in window]

                # Calculate the influence of each nearby chart (including the pivot).
                nearby_relation = [r.relation for r in rel_window]
                nearby_strength = [r.strength for r in rel_window]
                nearby_contrib = [r * s * v for r, s, v in zip(nearby_relation, nearby_strength, nearby_scobility)]

                # Replace the scobility with the window-averaged value.
                influence = sum(nearby_contrib) / sum(nearby_strength)
                self.scobility[i] = influence

            # Sort the chart list again based on the new scobility values.
            scobility_sort = [x for x in zip(self.scobility, self.ordering)]
            scobility_sort.sort(key=lambda x: x[0])
            self.scobility = [x[0] / scobility_sort[0][0] for x in scobility_sort]          # Keep minimum scobility pinned at 0.0
            self.ordering = [x[1] for x in scobility_sort]

            convergence.append(np.log2(self.scobility[-1]))
            if verbal:
                print(f'>>> Iteration {k+1:2d}: max scobility = {convergence[-1]:0.6f}')
                if False:
                    for i, (prev, next) in enumerate(zip(ordering_last, self.ordering)):
                        if prev.s_id != next.s_id:
                            print(f'Iteration {k+1:2d}... #{i:3d}: {prev} -> {next}')

        return convergence

    def view_scobility(self):
        for i, s in enumerate(self.ordering):
            print(f"{np.log2(self.scobility[i]):5.3f}🌶️ {str(s):>60s}")



def process_itl():
    itl = Tournament()

    # https://github.com/G-Wen/itl_history
    print('========================================================================')
    print('=== Loading data...')

    if not os.path.exists('itl_data'):
        raise FileNotFoundError('ITL tournament data scrape needs to be unpacked to /itl_data!')

    # Song info
    song_files = glob.glob('itl_data/song_info/*.json')
    for fn in song_files:
        with open(fn, 'r') as fp:
            song_data = json.load(fp)
            s = Song(song_data['song'])
            itl.songs[s.s_id] = s

    # Player info
    player_files = glob.glob('itl_data/entrant_info/*.json')
    for fn in player_files:
        with open(fn, 'r') as fp:
            player_data = json.load(fp)
            p = Player(player_data['entrant'])
            itl.players[p.e_id] = p

    # Score data
    score_files = glob.glob('itl_data/song_scores/*.json')
    for fn in score_files:
        with open(fn, 'r') as fp:
            score_data = json.load(fp)
            itl.load_score_data(score_data['scores'])

    # Calculate relationships
    print('========================================================================')
    print('=== Setting up relationships...')
    itl.setup_relationships()
    print('========================================================================')
    print('=== Calculating relationships...')
    itl.calc_relationships(verbal=False)
    print('========================================================================')
    print('=== Setting up closest-neighbor initial order...')
    itl.order_songs_initial(verbal=False, visual=False)
    print('========================================================================')
    print('=== Bubbling out non-monotonicities...')
    itl.order_refine_monotonic(verbal=True, visual=False)
    print('========================================================================')
    print('=== Refining scobility using neighborhood influence...')
    itl.order_refine_scobility(verbal=True, visual=False)
    print('========================================================================')
    print('=== Scobility calculation complete!...')
    itl.view_scobility()

    # Store (and re-load?)
    # with open('scobility_itl.json', 'w') as fp:
    #     json.dump(itl.dump(), fp, indent='\t')
    
    # with open('scobility_itl.json', 'r') as fp:
    #     itl2 = Tournament.load(json.load(fp))

    print('========================================================================')
    print('=== Done!')

    return itl


if __name__ == '__main__':
    process_itl()
    