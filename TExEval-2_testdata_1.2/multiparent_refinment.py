def delete_all_multiple_parents(G):
    for node in G.nodes():
        if G.in_degree(node) >= 5:
            edges_q = list(G.in_edges(node))
            for edge in edges_q:
                G.remove_edge(*edge)


def ppl_resolution(G, n):
    for node in G.nodes():
        if G.in_degree(node) >= n:
            edges_q = G.in_edges(node)
            ppls = {}
            for edge in edges_q:
                weight = G[edge[0]][edge[1]]["weight"]
                ppls[edge] = weight

            max_ppl_key = min(ppls, key=ppls.get)
            edges_q = list(edges_q)
            edges_q.remove(max_ppl_key)
            G.remove_edges_from(edges_q)
    return G


def synset_resolution(G, **kwargs):
    for node in G.nodes():
        if G.in_degree(node) >= kwargs["n"]:
            pairs = kwargs["helper"][node]
            for pair in pairs:
                if kwargs["ppl_compare"][pair] > kwargs["mix_thr"]:
                    parents = pair[0].split("_")
                    edge1, edge2 = (parents[0], pair[1]), (parents[1], pair[1])
                    if edge2 in G.in_edges(node) and edge1 in G.in_edges(node):
                        if (
                            G[parents[0]][node]["weight"]
                            > G[parents[1]][node]["weight"]
                        ):
                            G.remove_edge(*edge1)
                        else:
                            G.remove_edge(*edge2)


def resolve_multiple_parents(G, **kwargs):
    if kwargs["enable_mixing"]:
        synset_resolution(
            G,
            ppl_compare=kwargs["ppl_compare"],
            helper=kwargs["helper"],
            mix_thr=kwargs["mix_thr"],
            n=kwargs["n"],
        )
    else:
        ppl_resolution(G, kwargs["n"])


def helping_dict(compare):
    """
    new view for mixes dataset {node: pair_from_ppl_compare}
    """
    helper = {}

    for i in compare:
        if i[1] not in helper:
            helper[i[1]] = [i]
        else:
            helper[i[1]].append(i)
    return helper
