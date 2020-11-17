from traffic_equilibrium.pathdb import Item
from array import array


def test_pathdb(path_db):
    db = path_db
    key = array('I', [1, 2, 3, 4]).tobytes()
    value = array('I', [4, 5]).tobytes()
    v = db.get_py(key)
    assert not v.present
    db.set_py(key, value)
    v = db.get_py(key)
    assert not v.present
    db.flush()
    v = db.get_py(key)
    assert v.tobytes() == value
    db.set_py(array('I', [2,3,4]).tobytes(), array('I', [4]).tobytes())
    db.flush()
    print("iterating through the db")
    item = db.reset_cursor()
    while db.cursor_is_valid():
        db.next_item(item)
        i, k, v = item.tobytes()
        print(i, k, array('I', k), v, array('I', v))
    print("done")


