import configparser

import pyodbc
import pymysql
import psycopg2 as pg


def get_conn(database, db_conf):
    if database == "mssql":
        conn = pyodbc.connect(r'Driver=' + db_conf[database]["driver"] +
                              ';Server=' + db_conf[database]["server"] +
                              ';Database=' + db_conf[database]["database"] +
                              ';UID=' + db_conf[database]["user"] +
                              ';PWD=' + db_conf[database]["password"] + ';')

    elif database == "postgresql":
        conn = pg.connect(database=db_conf[database]["database"],
                          user=db_conf[database]["user"],
                          password=db_conf[database]["password"],
                          host=db_conf[database]["host"],
                          port=db_conf[database]["port"])

    elif database == "mysql":
        conn = pymysql.connect(host=db_conf[database]["host"],
                               user=db_conf[database]["user"],
                               password=db_conf[database]["password"],
                               database=db_conf[database]["database"],
                               port=int(db_conf[database]["port"]))

    return conn
