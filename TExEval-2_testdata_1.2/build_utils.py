from cycle_refinment import resolve_graph_cycles
from multiparent_refinment import resolve_multiple_parents
from conflict_refinment import refine_conflict
import networkx as nx
import numpy as np
from tqdm import tqdm


class TaxonomyBuilder:
    def __init__(self, root, all_verteces, max_iter=10000):
        self.root = root
        self.all_verteces = all_verteces
        self.max_iter = max_iter

    def build_taxonomy(self, strategy, **kwargs):
        self.edge_collector = getattr(self, strategy)
        self.collector_params = kwargs

        # self.pbar = tqdm(total=34000)
        self.all_edges = []
        self.i = 0
        self.build_tree(self.root, self.all_verteces)
        # self.pbar.close()

    #   return self.all_edges

    def build_tree(self, root, possible_verteces):
        top_edges_idx = self.edge_collector(
            root, possible_verteces, **self.collector_params
        )
        new_pos_verteces = np.delete(possible_verteces, top_edges_idx)
        for new_edge_idx in top_edges_idx:
            self.all_edges.append((root, possible_verteces[new_edge_idx]))
            # self.pbar.update(1)
            self.i += 1
            if self.i > self.max_iter:
                break
            self.build_tree(possible_verteces[new_edge_idx], new_pos_verteces)

    @staticmethod
    def ppl_thr_collector(root, possible_verteces, **kwargs):
        ppls = np.array(
            [kwargs["ppl_pairs"][(root, vertex)] for vertex in possible_verteces]
        )
        return np.where(np.array(ppls) < kwargs["thr"])[0]

    @staticmethod
    def ppl_top_collector(root, possible_verteces, **kwargs):
        ppls = np.array(
            [kwargs["ppl_pairs"][(root, vertex)] for vertex in possible_verteces]
        )
        return np.argsort(ppls)[: min(kwargs["top_k"], len(ppls))]


def iterative_child(ppl_pairs, low, high, step, max_iter):
    global G

    thrs = np.arange(low, high, step)
    Fs = []
    for thr in tqdm(thrs):
        tb = TaxonomyBuilder(root, all_verteces, max_iter)
        tb.build_taxonomy("ppl_thr_collector", ppl_pairs=ppl_pairs, thr=thr)
        edges = tb.all_edges

        P = len(set(G.edges()) & set(edges)) / (len(set(edges)) + 1)
        R = len(set(G.edges()) & set(edges)) / len(set(G.edges()))
        F = (2 * P * R) / (P + R + 1e-15)

        #  print('precision: {} \n recall: {} \n F-score: {}'.format(P,R,F))
        Fs.append(F)

    print(max(Fs), thrs[np.argmax(Fs)])
    return Fs


def brute_child(
    G,
    ppl_pairs,
    low,
    high,
    step,
    n,
    ppl_compare,
    helper,
    enable_mixing,
    insertions,
    insertions_conflict,
    ppls_hypo,
    use_insertion,
    resolve_conflicts,
):
    thrs = np.arange(low, high, step)
    mix_thrs = np.arange(0.5, 2, 0.1)
    if resolve_conflicts:
        conflict_thrs = np.linspace(1, 55, 40)
        if use_insertion:
            insertions_thrs = np.linspace(0.5, 4, 20)
        else:
            insertions_thrs = [1]
    else:
        conflict_thrs = [1]
        insertions_thrs = [1]

    Fs = []
    for thr in tqdm(thrs):
        for conflict_thr in conflict_thrs:
            for insert_thr in insertions_thrs:
                for mix_thr in mix_thrs:
                    G_pred = get_graph(ppl_pairs, thr)

                    if resolve_conflicts:
                        refine_conflict(
                            G_pred=G_pred,
                            ppls_hypo=ppls_hypo,
                            ppls_pairs=ppl_pairs,
                            conflict_thr=conflict_thr,
                            insertions_conflict=insertions_conflict,
                            use_insertion=use_insertion,
                            insert_thr=insert_thr,
                        )
                    resolve_graph_cycles(G_pred, insertions)
                    resolve_multiple_parents(
                        G_pred,
                        enable_mixing=enable_mixing,
                        ppl_compare=ppl_compare,
                        helper=helper,
                        mix_thr=mix_thr,
                        n=n,
                    )

                    P = len(set(G.edges()) & set(G_pred.edges())) / (
                        len(set(G_pred.edges())) + 1e-15
                    )
                    R = len(set(G.edges()) & set(G_pred.edges())) / len(set(G.edges()))
                    F = (2 * P * R) / (P + R + 1e-15)

                    Fs.append(F)

    print(max(Fs))
    return Fs


def get_graph(ppl_pairs, thr):
    S = nx.DiGraph()
    for key, val in ppl_pairs.items():
        if val < thr:
            S.add_edge(key[0], key[1], weight=val)
    return S


def clean_dict(pairs, use_lemma, reverse):
    new_pairs = {}
    for key, val in pairs.items():
        if use_lemma:
            term = key[0].split("(")[0].strip()
        else:
            term = key[0]
        target = key[1].split(",")[0]
        new_key = (target, term) if reverse else (term, target)
        new_pairs[new_key] = val

    return new_pairs
