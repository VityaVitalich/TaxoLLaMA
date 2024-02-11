import networkx as nx


def clean_triplets(triplets):
    new_triplets = {}

    for k, v in triplets.items():
        first_parent, second_parent = k[0].split("_")

        new_triplets[(first_parent, k[1][:-1], second_parent)] = v
    return new_triplets


def resolve_cycle_ppl(cur_G, cycle):
    cycle_ppls = {}

    for u, v in cycle:
        val = cur_G[u][v]["weight"]
        cycle_ppls[(u, v)] = val

    highest_ppl = sorted(cycle_ppls.items(), key=lambda x: x[1], reverse=True)[0][0]
    cur_G.remove_edge(*highest_ppl)


def simple_triplets_generator(G):
    """
    Generator function that returns triplets with condition: middle node has only one child
    """
    for node, degree in G.out_degree():
        if degree >= 1:
            for child in G.successors(node):
                yield (node, child, list(G.successors(child))[0])


def resolve_cycle_insertion(cur_G, cycle, insertions):
    temp_G = nx.DiGraph()
    for u, v in cycle:
        temp_G.add_node(u)
        temp_G.add_node(v)
        temp_G.add_edge(u, v)

    gen = simple_triplets_generator(temp_G)

    triplets_ppl = {}
    for triplet in gen:
        triplets_ppl[triplet] = insertions[triplet]

    highest_ppl_triplet = max(triplets_ppl, key=triplets_ppl.get)
    grand, parent, child = highest_ppl_triplet
    if cur_G[grand][parent]["weight"] > cur_G[parent][child]["weight"]:
        cur_G.remove_edge(grand, parent)
    else:
        cur_G.remove_edge(parent, child)


def resolve_graph_cycles(G_pred, insertions):
    while True:
        try:
            cycle = nx.find_cycle(G_pred)
            if len(cycle) > 2:
                # print(0)
                try:
                    resolve_cycle_insertion(G_pred, cycle, insertions)
                    # resolve_cycle_ppl(G_pred, cycle)
                except KeyError:
                    resolve_cycle_ppl(G_pred, cycle)
            else:
                resolve_cycle_ppl(G_pred, cycle)
        except nx.NetworkXNoCycle:
            break
