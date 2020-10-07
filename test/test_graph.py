from traffic_equilibrium.graph import DiGraph


def test_init():
    g = DiGraph()
    assert g.name == ""
    info = g.info()
    assert info.name == ""
    assert info.number_of_nodes == 0
    assert info.number_of_links == 0
    assert g.links() == []


def test_add_nodes():
    g = DiGraph()
    assert g.info().number_of_nodes == 0
    g.append_nodes(5)
    assert g.info().number_of_nodes == 5
    g.append_nodes(5)
    assert g.info().number_of_nodes == 10
    assert g.links() == []


def test_add_links():
    g = DiGraph()
    g.append_nodes(4)
    links_to_add = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    g.add_links_from(links_to_add)
    info = g.info()
    assert info.number_of_nodes == 4
    assert info.number_of_links == 5
    assert g.links() == links_to_add
