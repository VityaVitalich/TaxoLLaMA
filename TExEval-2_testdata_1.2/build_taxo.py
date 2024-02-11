import yaml
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx
from build_utils import clean_dict, brute_child, iterative_child
from multiparent_refinment import helping_dict
from cycle_refinment import clean_triplets
from conflict_refinment import clean_triplets_conflicts

with open(r"./configs/build_taxo.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)


if __name__ == "__main__":
    data = params_list["DATA"][0]
    in_name = params_list["IN_NAME"][0]
    enable_mixing = False
    lemma = params_list["LEMMA"][0]
    reverse = params_list["REVERSE"][0]
    n = params_list["N_PARENTS"][0]
    low = params_list["LOW"][0]
    high = params_list["HIGH"][0]
    step = params_list["STEP"][0]
    resolve_conflicts = False
    use_insertion = False

    if data == "food":
        path = "gs_taxo/EN/" + str(data) + "_wordnet_en.taxo"
    else:
        path = "gs_taxo/EN/" + str(data) + "_eurovoc_en.taxo"
    G = nx.DiGraph()

    with open(path, "r") as f:
        for line in f:
            idx, hypo, hyper = line.split("\t")
            hyper = hyper.replace("\n", "")
            G.add_node(hypo)
            G.add_node(hyper)
            G.add_edge(hyper, hypo)

    with open(in_name, "rb") as f:
        ppls = pickle.load(f)

    ppls_pairs = clean_dict(ppls, use_lemma=lemma, reverse=reverse)


    root = data
    all_verteces = list(G.nodes)
    all_verteces.remove(root)

    #  print(ppls_pairs)
    build_args = {
        "G": G,
        "ppl_pairs": ppls_pairs,
        "low": low,
        "high": high,
        "step": step,
    }

    mix_parents_args = {
        "ppl_compare": None,
        "helper": None,
        "enable_mixing": False,
        "n": n,
    }

    cycles_args = {
        "insertions": None,
    }

    conflict_args = {
        "resolve_conflicts": resolve_conflicts,
        "insertions_conflict": None,
        "ppls_hypo": None,
        "use_insertion": use_insertion,
    }

    build_args.update(mix_parents_args)
    build_args.update(cycles_args)
    build_args.update(conflict_args)

    res = brute_child(**build_args)
