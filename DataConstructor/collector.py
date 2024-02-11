from typing import Dict, Generator, Optional, Set, Tuple

import networkx as nx

import numpy as np
from copy import deepcopy


class GeneratorMixin:
    def leafs_generator(self) -> Generator[Tuple[str, Dict[int, Set[str]]], None, None]:
        for node, degree in self.G.out_degree():
            if (
                degree == 0
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                parents = self.find_parents(node, self.ancestors_depth)

                yield ((node, parents))

    def parent_generator(
        self,
    ) -> Generator[Tuple[str, Dict[int, Set[str]]], None, None]:
        for node, degree in self.G.out_degree():
            if (
                degree > 0
                and self.generations[node] > self.generation_depth
                and len(node) > 1
                and degree < 50
            ):
                # parents = self.find_parents(node, self.ancestors_depth)
                for child in self.G.successors(node):
                    yield ((node, child))

    def only_child_generator(self) -> None:
        """
        Generator function that return verteces that has only one leaf children
        """
        for node, degree in self.G.out_degree():
            if (
                degree == 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                for child in self.G.successors(node):
                    if self.G.out_degree(child) == 0:
                        for grandparent in self.G.predecessors(node):
                            yield (child, node, grandparent)

    def all_children_leafs_generator(self):
        """
        Generator function that returns vertices that are all children of one parent
        and they are all leafs
        """
        for node, degree in self.G.out_degree():
            if (
                degree > 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                all_children_leafs = True
                for child in self.G.successors(node):
                    if self.G.out_degree(child) > 0:
                        all_children_leafs = False
                        break

                if all_children_leafs:
                    yield (node, list(self.G.successors(node)))

    def leaf_and_no_leaf_generator(self):
        """
        Generator function that returns verteces that are leafs but other children
        of their parents are not leafs
        """
        for node, degree in self.G.out_degree():
            if (
                degree > 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                all_children_leafs = True
                exists_a_leaf = False
                for child in self.G.successors(node):
                    if self.G.out_degree(child) > 0:
                        all_children_leafs = False
                    if self.G.out_degree(child) == 0 and not all_children_leafs:
                        exists_a_leaf = True
                        break

                if all_children_leafs or not exists_a_leaf:
                    continue
                else:
                    leafs = []
                    non_leafs = []
                    for child in self.G.successors(node):
                        if self.G.out_degree(child) == 0:
                            leafs.append(child)
                        else:
                            non_leafs.append(child)
                    yield (leafs, non_leafs, node)

    def simple_triplets_generator(self):
        """
        Generator function that returns triplets with condition: middle node has only one child
        """
        for node, degree in self.G.out_degree():
            if (
                degree >= 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                for child in self.G.successors(node):
                    if self.G.out_degree(child) == 1:
                        yield (node, child, list(self.G.successors(child))[0])

    def mixes_generator(self):
        """
        Generator function that returns triplets with 2 parents with their common child
        """
        for node, degree in self.G.out_degree():
            if self.generations[node] > self.generation_depth and len(node) > 1:
                parents = list(self.G.predecessors(node))
                if len(parents) > 1:
                    for i in range(len(parents)):
                        for j in range(i + 1, len(parents)):
                            yield (node, parents[i], parents[j])


class Collector(GeneratorMixin):
    def __init__(
        self,
        G,
        generation_depth: int = 40,
        ancestors_depth: int = 2,
        p_test=0.1,
        p_divide_leafs=0.5,
        min_to_test_rate=0.5,
        weights=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        use_1sense_hypernyms=True,
    ):
        self.generation_depth = generation_depth
        self.ancestors_depth = ancestors_depth
        self.p = p_test
        self.p_divide_leafs = p_divide_leafs
        self.min_to_test_rate = min_to_test_rate
        self.G = G
        self.weights = weights
        self.use_1sense_hypernyms = use_1sense_hypernyms

        self.train = []
        self.test = []
        self.test_verteces = set()
        self.train_verteces = set()

        self.precompute_generations()

        self.node_numbers = {}
        for node in G.nodes():
            node_name = node.split(".")[0]
            try:
                node_number = int(node.split(".")[-1])
            except ValueError:
                print(node)
                continue

            if node_name in self.node_numbers.keys():
                self.node_numbers[node_name] = max(
                    self.node_numbers[node_name], node_number
                )
            else:
                self.node_numbers[node_name] = node_number

    def precompute_generations(self) -> None:
        self.generations = {}
        for i, gen in enumerate(nx.topological_generations(self.G)):
            for word in gen:
                self.generations[word] = i

    def init_generators(self) -> None:
        self.gen_only_child = self.only_child_generator()
        self.gen_only_leafs = self.all_children_leafs_generator()
        self.gen_not_only_leafs = self.leaf_and_no_leaf_generator()
        self.mixes_triplets = self.mixes_generator()
        self.simple_triplets = self.simple_triplets_generator()
        self.gen_parent = self.parent_generator()

        self.all_generators = [
            "gen_only_child",
            "gen_only_leafs",
            "gen_not_only_leafs",
            "mixes_triplets",
            "simple_triplets",
            "gen_parent",
        ]

        self.gen2func = {
            "gen_only_child": self.collect_sample_child,
            "gen_only_leafs": self.collect_only_leafs,
            "gen_not_only_leafs": self.collect_not_only_leafs,
            "mixes_triplets": self.collect_mixes_triplets,
            "simple_triplets": self.collect_simple_triplets,
            "gen_parent": self.collect_parents,
        }

    def collect_possible_samples(self):
        self.init_generators()

        active_generators = []
        active_weights = []
        for i, weight in enumerate(self.weights):
            if weight > 0:
                active_generators.append(self.all_generators[i])
                active_weights.append(weight)

        while active_generators:
            sample = np.random.choice(self.all_generators, p=self.weights)
            if sample in active_generators:
                try:
                    self.gen2func[sample]()
                except StopIteration:
                    active_generators.remove(sample)

    def collect_sample_child(self):
        child, parent, grandparent = next(self.gen_only_child)
        elem = {}
        elem["children"] = child
        elem["parents"] = parent
        elem["grandparents"] = grandparent
        elem["case"] = "only_child_leaf"

        self.simple_filter(elem)

    def collect_only_leafs(self):
        parent, children = next(self.gen_only_leafs)
        elem = {}
        elem["children"] = children
        elem["parents"] = parent
        elem["grandparents"] = None
        elem["case"] = "only_leafs_all"

        if len(elem["children"]) > 50:
            return
        possible_test, possible_train = self.get_possible_train(children)
        to_test_rate = len(possible_test) / len(children)

        if to_test_rate == 1:
            if self.goes_to_test():
                if self.divide_on_half():
                    elem["case"] = "only_leafs_divided"
                    to_test_subset = children[: len(children) // 2]
                    to_train_subset = children[len(children) // 2 :]

                    elem["brothers"] = to_train_subset
                    self.write_to_test(elem, to_test_subset)

                    elem["case"] = "only_leafs_all"
                    self.write_to_train(
                        elem, to_train_subset
                    )  

                else:
                    self.write_to_test(elem, children)

            else:
                if self.divide_on_half():
                    elem["case"] = "only_leafs_divided"
                    # делим пополам
                    to_test_subset = children[: len(children) // 2]
                    to_train_subset = children[len(children) // 2 :]

                    elem["brothers"] = to_train_subset
                    self.write_to_train(
                        elem, to_test_subset
                    )  
                else:
                    self.write_to_train(elem, children)

        elif to_test_rate >= self.min_to_test_rate:
            if self.goes_to_test():
                elem["case"] = "only_leafs_divided"

                to_obtain_half = (len(children) // 2) - len(possible_train)
                possible_train = possible_train + possible_test[:to_obtain_half]
                possible_test = possible_test[to_obtain_half:]

                elem["brothers"] = possible_train
                self.write_to_test(elem, possible_test)

                elem["case"] = "only_leafs_all"
                self.write_to_train(elem, possible_train)

            else:
                if self.divide_on_half():

                    elem["case"] = "only_leafs_divided"
                    to_test_subset = children[: len(children) // 2]
                    to_train_subset = children[len(children) // 2 :]

                    elem["brothers"] = to_train_subset
                    self.write_to_train(
                        elem, to_test_subset
                    )  
                else:
                    self.write_to_train(elem, children)
        else:
            self.write_to_train(elem, children)

    def collect_not_only_leafs(self):
        children_leafs, children_no_leafs, parent = next(self.gen_not_only_leafs)
        elem = {}
        elem["children"] = children_leafs + children_no_leafs
        elem["parents"] = parent
        elem["grandparents"] = None
        elem["case"] = "leafs_and_no_leafs"

        if len(elem["children"]) > 50:
            return

        # if (
        #     (len(elem["children"]) < 50)
        #     and self.predict_parent()
        #     and (self.node_numbers[parent.split(".")[0]] == 1)
        # ):
        #     elem["case"] = "predict_hypernym"
        #     for child in elem["children"]:
        #         elem["children"] = child
        #         self.simple_filter(elem)
        #     return

        possible_test, possible_train = self.get_possible_train(children_leafs)
        to_test_rate = len(possible_test) / len(children_leafs)
        # print(to_test_rate)
        if to_test_rate == 1:
            if self.goes_to_test():
                self.write_to_test(elem, possible_test)
                self.write_to_train(elem, children_no_leafs)
            else:
                self.write_to_train(elem, children_leafs + children_no_leafs)

        else:
            self.write_to_train(elem, children_leafs + children_no_leafs)

    def collect_simple_triplets(self):
        grandparent, parent, child = next(self.simple_triplets)
        elem = {}
        elem["children"] = child
        elem["parents"] = parent
        elem["grandparents"] = grandparent
        elem["case"] = "simple_triplet_grandparent"

        self.simple_filter(elem)

    def collect_mixes_triplets(self):
        child, parent1, parent2 = next(self.mixes_triplets)
        elem = {}
        elem["children"] = child
        elem["parents"] = [parent1, parent2]
        elem["grandparents"] = None
        elem["case"] = "simple_triplet_2parent"

        self.simple_filter(elem)

    def collect_parents(self):
        parent, child = next(self.gen_parent)
        try:
            num_senses = self.node_numbers[child.split(".")[0]]
        except KeyError:
            num_senses = 2

        if self.use_1sense_hypernyms:
            if num_senses == 1:
                elem = {}
                elem["children"] = child
                elem["parents"] = parent
                elem["grandparents"] = None
                elem["case"] = "predict_hypernym"
                self.simple_filter(elem)
        else:
            elem = {}
            elem["children"] = child
            elem["parents"] = parent
            elem["grandparents"] = None
            elem["case"] = "predict_hypernym"
            self.simple_filter(elem)

    def simple_filter(self, elem):
        if elem["children"] in self.train_verteces:
            self.train.append(elem)
        else:
            if elem["children"] in self.test_verteces:
                self.test.append(elem)
            else:
                if self.goes_to_test():
                    self.test.append(elem)
                    self.test_verteces.add(elem["children"])
                else:
                    self.train.append(elem)
                    self.train_verteces.add(elem["children"])

    def get_possible_train(self, children):
        """
        Returns two lists that could possibly go to test
        and that can not. It is checked through set of train verteces
        """
        possible_test = []
        possible_train = []
        for child in children:
            if child in self.train_verteces:
                possible_train.append(child)
            else:
                possible_test.append(child)

        return possible_test, possible_train

    def goes_to_test(self):
        return np.random.binomial(1, p=self.p)

    def divide_on_half(self):
        return np.random.binomial(1, p=self.p_divide_leafs)

    def predict_parent(self):
        return np.random.binomial(1, p=self.p_parent)

    def write_to_test(self, elem, to_test_subset):
        """
        writes to test a subset
        """
        elem = deepcopy(elem)
        elem["children"] = to_test_subset
        for vertex in to_test_subset:
            self.test_verteces.add(vertex)
        self.test.append(elem)

    def write_to_train(self, elem, to_train_subset):
        """'
        writes to train verteces that are not in test
        """
        elem = deepcopy(elem)
        valid_verteces = []
        for vertex in to_train_subset:
            if vertex not in self.test_verteces:
                self.train_verteces.add(vertex)
                valid_verteces.append(vertex)

        if valid_verteces:
            elem["children"] = valid_verteces
            self.train.append(elem)
