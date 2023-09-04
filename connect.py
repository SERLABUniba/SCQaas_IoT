#!/usr/bin/python3
import psycopg2
from config import config
import numpy as np

from datetime import datetime


def connect(sql, values):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        # create a cursor
        cur = conn.cursor()
        # execute a statement
        #print('Postgresql database version:')
        #cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        #db_version = cur.fetchone()
        #print(db_version)

        ###################################################
        # Use Executemany to insert many rows at once
        # list of rows to be inserted
        #values = [(17, 'rachel', 67), (18, 'ross', 79), (19, 'nick', 95)]
        #
        # executing the sql statement
        #cursor.executemany("INSERT INTO classroom VALUES(%s,%s,%s)", values)
        #
        ###################################################

        if len(values.shape) == 1:
            print('1-D array')
            # Update table with requested data
            cur.execute(sql, (str(values[0]).strip(), str(values[1]).strip(), str(values[2]).strip(),
                              str(values[3]).strip(), str(values[4]).strip(), str(values[5]).strip()))
        elif len(values.shape) == 2:
            cur.executemany(sql, values)

        #cur.execute("insert into scmobility.test values(%s, %s);", (5, 's4 text'))
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


if __name__ == '__main__':
    connect()
