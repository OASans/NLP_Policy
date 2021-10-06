import sqlite3
import pandas as pd


class PolicyDB:
    def __init__(self, db_name='policy.db'):
        self.conn = sqlite3.connect(db_name)

    def display_all_tables(self):
        c = self.conn.cursor()
        c.execute("select name from sqlite_master where type='table' order by name")
        print(c.fetchall())

    def create_table(self, sql):
        # sql = """create table tagger (
        # id varchar(20),
        # name varchar(20) primary key,
        # status varchar(10)
        # )"""
        c = self.conn.cursor()
        c.execute(sql)
        self.conn.commit()

    def drop_table(self, table_name):
        c = self.conn.cursor()
        c.execute("""drop table {}""".format(table_name))
        self.conn.commit()

    def select_from_table(self, table_name="", sql=""):
        c = self.conn.cursor()
        if sql != "":
            c.execute(sql)
        else:
            c.execute("""select * from {}""".format(table_name))
        print(c.fetchall())

    def delete_from_table(self, table_name="", sql=""):
        c = self.conn.cursor()
        if sql!= "":
            c.execute(sql)
        else:
            c.execute("""delete from {} where 1=1""".format(table_name))
        self.conn.commit()

    def update_tagger_info(self):
        people = pd.DataFrame(
            [['王昕', '20307090013', 1], ['虎雪', '19307090201', 0], ['苏慧怡', '19307090197', 1], ['赵子昂', '20307090056', 1], ['王心恬', '19307090134', 1],
             ['冷方琼', '20307090139', 1], ['周人芨', '19307090009', 1], ['南楠', '20307090155', 1], ['姜静宜', '19307090065', 0], ['陈奔逸', '19307090045', 1],
             ['苏品菁', '20307090170', 1], ['王远', '20307090213', 0], ['张宇恬', '19307090122', 1], ['初子菡', '18307090138', 1],
             ['陈涛', '19307090052', 0], ['严可欣', '19307090078', 0], ['徐雪', '21210170019', 0], ['柏柯羽', '20307090147', 0],
             ['帕提曼', '19307090210', 1], ['贺凡熙', '21210170020', 1], ['施烨昕', '', 1], ['蒋佳钰', '', 1]], columns=['name', 'id', 'status'])
        people.to_sql('tagger', self.conn, if_exists='replace', index=False)
        self.conn.commit()

    def count_tagger(self):
        c = self.conn.cursor()
        c.execute("""select count(name), * from tagger where status==1""")
        print(c.fetchall())

