from argparse import ArgumentParser
from array import array

from traffic_equilibrium.pathdb import PathDB


def main(dbname):
    db = PathDB(dbname)
    cursor = db.cursor()
    cursor.reset()
    while cursor.is_valid():
        cursor.populate()
        k = cursor.key()
        v = cursor.value()
        print(cursor.counter, k, array('I', k), v, array('I', v))
        if not db.get_py(k).present:
            print(f"Key NOT FOUND: {k}")
        cursor.next()

parser = ArgumentParser()
parser.add_argument('path')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.path)
