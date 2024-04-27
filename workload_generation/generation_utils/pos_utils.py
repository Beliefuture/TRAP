from workload_generation.generation_utils.db_con import get_conn
from workload_generation.generation_utils.com_utils import get_conf


class PostgreUtils:
    def __init__(self, conf_file, database="postgresql"):
        db_conf = get_conf(conf_file)

        self.conn = get_conn(database, db_conf)
        self.cursor = self.conn.cursor()

        self.conn.autocommit = True

    def exec_fetch(self, statement, one=True):
        self.cursor.execute(statement)
        if one:
            return self.cursor.fetchone()
        return self.cursor.fetchall()

    def exec_only(self, statement):
        self.cursor.execute(statement)

    def create_statistics(self, value=0.17):
        self.exec_only(f"SELECT setseed({value})")
        self.exec_only("analyze")

    def create_hypo_index(self, index):
        for ind in index:
            ind_name = f"{ind[0]}_{'_'.join(ind[1])}_idx"
            statement = (
                "select * from hypopg_create_index( "
                f"'create index {ind_name} on {ind[0]} "
                f"({','.join(ind[1])})')"
            )
            print(statement+";")
            self.exec_only(statement)

    def drop_hypo_index(self):
        stmt = "SELECT * FROM hypopg_reset();"
        self.exec_only(stmt)

    def create_index(self, index):
        index_info = []
        for ind in index:
            ind_name = f"{ind[0]}_{'_'.join(ind[1])}_idx"
            statement = (
                f"create index {ind_name} "
                f"on {ind[0]} ({','.join(ind[1])})"
            )
            self.exec_only(statement)
            statement = (
                f"select relpages from pg_class c " f"where c.relname = '{ind_name}'"
            )
            size = self.exec_fetch(statement)
            size = size[0]
            index_info.append((ind_name, size * 8 * 1024))

        return index_info

    def drop_index(self):
        stmt = "select indexname from pg_indexes where schemaname='public'"
        indexes = self.exec_fetch(stmt, one=False)
        for index in indexes:
            index_name = index[0]
            if "_pkey" not in index_name:
                drop_stmt = "drop index {}".format(index_name)
                self.exec_only(drop_stmt)

    def get_est_cost(self, query):
        # create view and return the next sql.
        # query_text = self._prepare_query(query)
        statement = f"explain (format json) {query}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        # drop view
        # self._cleanup_query(query)
        total_cost = query_plan["Total Cost"]

        return total_cost

    def create_indexes(self, indexes, mode="hypo"):
        try:
            for index in indexes:
                index_def = index.split("#")
                index_name = index.replace("#", "_").replace(",", "_")
                stmt = f"create index on {index_def[0]} ({index_def[1]})"
                if len(index_def) == 3:
                    stmt += f" include ({index_def[2]})"
                if mode == "hypo":
                    stmt = f"select * from hypopg_create_index('{stmt}')"
                self.exec_only(stmt)
        except Exception as e:
            print(e)
            print(stmt)

    def drop_indexes(self, mode):
        if mode == "hypo":
            stmt = "select * from hypopg_reset();"
            self.exec_only(stmt)
        else:
            stmt = "select indexname from pg_indexes where schemaname='public'"
            indexes = self.exec_fetch(stmt, one=False)
            for index in indexes:
                index_name = index[0]
                if "_pkey" not in index_name:
                    drop_stmt = "drop index {}".format(index_name)
                    self.exec_only(drop_stmt)

    def get_ind_cost(self, query, indexes, mode="hypo"):
        self.create_indexes(indexes, mode)

        stmt = f"explain (format json) {query}"
        query_plan = self.exec_fetch(stmt)[0][0]["Plan"]
        # drop view
        # self._cleanup_query(query)
        total_cost = query_plan["Total Cost"]

        self.drop_indexes(mode)

        return total_cost

    def get_ind_cost_plan(self, query, indexes, mode="hypo"):
        self.create_indexes(indexes, mode)

        stmt = f"explain (format json) {query}"
        query_plan = self.exec_fetch(stmt)[0][0]["Plan"]
        # drop view
        # self._cleanup_query(query)
        total_cost = query_plan["Total Cost"]

        self.drop_indexes(mode)

        return total_cost, query_plan

    def get_tables(self):
        tables = []
        sql = "select * from pg_tables where schemaname = 'public';"
        rows = self.exec_fetch(sql, one=False)
        for row in rows:
            tables.append(row[1])

        return tables

    def get_cols(self, table):
        cols = []
        sql = f"select column_name, data_type from information_schema.columns where " \
              f"table_schema='public' and table_name='{table}'"

        rows = self.exec_fetch(sql, one=False)
        for row in rows:
            cols.append(row)

        return cols

    def get_row_count(self, tables):
        rows = []
        for t in tables:
            sql = f"SELECT COUNT(*) FROM {t}"
            rows.append((t, self.exec_fetch(sql, one=True)[0]))

        return rows

    def get_selectivity(self):
        i = 0
        sels = dict()

        tables = self.get_tables()
        for table in tables:
            cols = self.get_cols(table)
            for col in cols:
                sql = f"select distinct {col}/count(*) from {table}"
                sel = self.exec_fetch(sql)[0]
                sels[col] = (i, sel)
                i += 1

        return sels

    def get_sample_data(self, sample_seed=0.01, min_row=50):
        sample_data = dict()
        tables = self.get_tables()
        for table in tables:
            for col in self.get_cols(table):
                sample_data[col[0]] = []
                sql = f"SELECT {col[0]} FROM {table} TABLESAMPLE BERNOULLI ({sample_seed});"
                rows = self.exec_fetch(sql, one=False)
                if len(rows) < min_row:
                    sql = f"SELECT {col[0]} FROM {table} ORDER BY random() LIMIT {min_row};"
                    rows = self.exec_fetch(sql, one=False)
                for row in rows:
                    if " " in str(row[0]):
                        sample_data[col[0]].append(row[0].strip(" "))
                    else:
                        sample_data[col[0]].append(row[0])

        return sample_data

    def get_query_plan(self, query):
        statement = f"explain (format json) {query}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]

        return query_plan
