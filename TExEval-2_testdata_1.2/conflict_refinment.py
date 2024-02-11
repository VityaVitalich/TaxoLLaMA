def clean_triplets_conflicts(triplets):
    new_triplets = {}

    for k, v in triplets.items():
        first_parent, second_parent = k[0].split("_")

        new_key = (first_parent, second_parent)
        if new_key in new_triplets.keys():
            new_triplets[new_key][k[1][:-1]] = v
        else:
            new_triplets[new_key] = {k[1][:-1]: v}

    return new_triplets


def conflict_refinment_ppl(G_pred, ppls_hypo, ppls_pairs, conflict_thr):
    edges = list(G_pred.edges())
    for u, v in edges:
        diff = abs(ppls_pairs[(u, v)] - ppls_hypo[(u, v)])

        if diff > conflict_thr:
            # print(diff)
            G_pred.remove_edge(u, v)


def conflict_refinment_insertion(
    G_pred, ppls_hypo, ppls_pairs, conflict_thr, insertions_conflict, insert_thr
):
    edges = list(G_pred.edges())
    for u, v in edges:
        diff = abs(ppls_pairs[(u, v)] - ppls_hypo[(u, v)])

        if diff > conflict_thr:
            try:
                best_middle_node, best_ppl = min(
                    insertions_conflict[(u, v)].items(), key=lambda x: x[1]
                )
                if best_ppl < insert_thr:
                    G_pred.add_edge(
                        u, best_middle_node, weight=ppls_pairs[(u, best_middle_node)]
                    )
                    G_pred.add_edge(
                        best_middle_node, v, weight=ppls_pairs[(best_middle_node, v)]
                    )

                # print('success')
            except KeyError:
                continue


def refine_conflict(
    G_pred,
    ppls_hypo,
    ppls_pairs,
    conflict_thr,
    insertions_conflict,
    use_insertion,
    insert_thr,
):
    if use_insertion:
        conflict_refinment_insertion(
            G_pred,
            ppls_hypo=ppls_hypo,
            ppls_pairs=ppls_pairs,
            conflict_thr=conflict_thr,
            insertions_conflict=insertions_conflict,
            insert_thr=insert_thr,
        )
    else:
        conflict_refinment_ppl(
            G_pred,
            ppls_hypo=ppls_hypo,
            ppls_pairs=ppls_pairs,
            conflict_thr=conflict_thr,
        )
