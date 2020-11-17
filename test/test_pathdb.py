from array import array


def test_pathdb(path_db):
    db = path_db
    key = array('I', [1, 2, 3, 4]).tobytes()
    value = array('I', [4, 5]).tobytes()
    v = db.get_py(key)
    assert not v.present
    db.set_py(key, value)
    v = db.get_py(key)
    assert v.tobytes() == value
    db.set_py(array('I', [2,3,4]).tobytes(), array('I', [4]).tobytes())
    print("iterating through the db")
    cursor = db.cursor()
    while cursor.is_valid():
        cursor.populate()
        i = cursor.counter
        k = cursor.key()
        v = cursor.value()
        print(i, k, array('I', k), v, array('I', v))
        cursor.next()
    print("done")


