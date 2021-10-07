import os
import re
import glob
import json
import shutil
import sqlite3
import collections
import numpy as np
import pandas as pd


class DBConfig:
    def __init__(self):
        self.add_new_data_to_db = False
        self.convert_db_to_dataset = True


class PolicyDB:
    def __init__(self, db_name='policy.db'):
        self.conn = sqlite3.connect(db_name)

    # 关闭数据库连接，最好在使用对象的最后调用一下
    def close_db(self):
        self.conn.close()

    # 展示数据库内所有表格
    def display_all_tables(self):
        c = self.conn.cursor()
        c.execute("select name from sqlite_master where type='table' order by name")
        print(c.fetchall())

    # 创建表
    def create_table(self, sql):
        # sql = """create table tagger (
        # id varchar(20),
        # name varchar(20) primary key,
        # status varchar(10)
        # )"""
        c = self.conn.cursor()
        c.execute(sql)
        self.conn.commit()

    # drop表
    def drop_table(self, table_name):
        c = self.conn.cursor()
        c.execute("""drop table {}""".format(table_name))
        self.conn.commit()

    # 从表中选择，只给表名的话默认选择全部记录
    def select_from_table(self, table_name="", sql=""):
        c = self.conn.cursor()
        if sql != "":
            c.execute(sql)
        else:
            c.execute("""select * from {}""".format(table_name))
        print(c.fetchall())

    # 从表中删除，只给表名的话默认删除全部记录
    def delete_from_table(self, table_name="", sql=""):
        c = self.conn.cursor()
        if sql!= "":
            c.execute(sql)
        else:
            c.execute("""delete from {} where 1=1""".format(table_name))
        self.conn.commit()

    # 更新标注人员信息
    def update_tagger_info(self):
        people = pd.DataFrame(
            [['王昕', '20307090013', 1], ['虎雪', '19307090201', 0], ['苏慧怡', '19307090197', 1], ['赵子昂', '20307090056', 1], ['王心恬', '19307090134', 1],
             ['冷方琼', '20307090139', 1], ['周人芨', '19307090009', 1], ['南楠', '20307090155', 1], ['姜静宜', '19307090065', 0], ['陈奔逸', '19307090045', 1],
             ['苏品菁', '20307090170', 1], ['王远', '20307090213', 0], ['张宇恬', '19307090122', 1], ['初子菡', '18307090138', 1],
             ['陈涛', '19307090052', 0], ['严可欣', '19307090078', 0], ['徐雪', '21210170019', 0], ['柏柯羽', '20307090147', 0],
             ['帕提曼', '19307090210', 1], ['贺凡熙', '21210170020', 1], ['施烨昕', '', 1], ['蒋佳钰', '', 1]], columns=['name', 'id', 'status'])
        people.to_sql('tagger', self.conn, if_exists='replace', index=False)
        self.conn.commit()

    # 查看目前在工作的标注人员
    def count_tagger(self):
        c = self.conn.cursor()
        c.execute("""select count(name) from tagger where status==1""")
        print(c.fetchall())
        c.execute("""select * from tagger where status==1""")
        print(c.fetchall())

    # 创建policy表
    def create_policy_table(self):
        # if policy already exists
        self.drop_table('policy')
        self.create_table("""create table policy (
                            uid varchar(20) primary key,
                            origin_id varchar(20), 
                            policy_name varchar(20),
                            annotate_status int,
                            tagger_name varchar(20),
                            annotate_time varchar(20)
                            )""")

    # 插入policy记录
    def insert_policies(self):
        cursor = self.conn.cursor()
        folder_list = glob.glob('/Volumes/TOURO Mobil/policy标注/政策文件rawdata/*')
        for folder in folder_list:
            uid = folder.split('/')[-1]
            origin_file_list = glob.glob('/Volumes/TOURO Mobil/policy标注/政策文件rawdata/{}/*'.format(uid))
            origin_file_list = [file for file in origin_file_list if
                                (file.split('.')[-1] != 'txt' and file.split('.')[-1] != 'xlsx')]
            if len(origin_file_list) != 1:
                print(uid)
            else:
                file_name = origin_file_list[0].split('/')[-1]
                origin_id = re.split('[._]', file_name)[-2]
                cursor.execute("INSERT INTO policy (uid,origin_id,policy_name,annotate_status,tagger_name, annotate_time) \
                              VALUES ('{}', '{}', '{}', 0, '', '')".format(uid, origin_id, file_name))
        self.conn.commit()

    # 创建annotated_sentence表
    def create_annotated_sentence_table(self):
        # if annotated_sentence already exists
        self.drop_table('annotated_sentence')
        self.create_table("""create table annotated_sentence (
                            uid varchar(20),
                            sid varchar(20) primary key,
                            
                            sentence text,
                            sentence_type varchar(20)
                            )""")

    # 创建entity表
    def create_entity_table(self):
        # if entity already exists
        self.drop_table('entity')
        self.create_table("""create table entity (
                            sid varchar(20),
                            entity text,
                            entity_type varchar(20)
                            )""")

    # 创建entry表
    def create_entry_table(self):
        # if entry already exists
        self.drop_table('entry')
        self.create_table("""create table entry (
                            sid varchar(20),
                            eid varchar(20),
                            entry_type varchar(20),
                            var text,
                            relation varchar(20),
                            field text,
                            var_context text,
                            field_context text,
                            primary key (sid,eid)
                            )""")

    # 创建entry_logic表
    def create_entry_logic_table(self):
        # if entry_logic already exists
        self.drop_table('entry_logic')
        self.create_table("""create table entry_logic (
                            uid varchar(20),
                            logic varchar(255),
                            logic_name varchar(255)
                            )""")

    # 更新policy表状态
    # info = (annotate_status, tagger_name, annotate_time, uid)
    def update_policy(self, info):
        sql = ''' UPDATE policy
                  SET annotate_status = ? ,
                      tagger_name = ? ,
                      annotate_time = ?
                  WHERE uid = ?'''
        cur = self.conn.cursor()
        cur.execute(sql, info)
        self.conn.commit()

    # 插入annotated_sentence表记录
    # info = (uid, sid, sentence, sentence_type)
    def insert_annotated_sentence(self, info):
        sql = ''' INSERT INTO annotated_sentence(uid,sid,sentence,sentence_type)
                  VALUES(?,?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, info)
        self.conn.commit()

    # 插入entity表记录
    # info = (sid, entity, entity_type)
    def insert_entity(self, info):
        # 时间格式问题，放到db2dataset中去处理
        # if info[2] == '发布时间' and info[1] not in sentence:
        #     time = str(info[1])
        #     time_match = re.search(r"(\d{4}\S\d{1,2}\S\d{1,2})", time)
        #     if not time_match:
        #         return
        #     time_pattern = time_match.group(0)[:4] + "\S\d{1,2}\S\d{1,2}号*日*"
        #     standard_time = re.search(time_pattern, sentence)
        #     if not standard_time:
        #         return
        #     info = (info[0], standard_time.group(0), info[2])
        sql = ''' INSERT INTO entity(sid,entity,entity_type)
                  VALUES(?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, info)
        self.conn.commit()

    # 插入entry表记录
    # info = (sid, eid, entry_type, var, relation, field, var_context, field_context)
    def insert_entry(self, info):
        sql = ''' INSERT INTO entry(sid, eid, entry_type, var, relation, field, var_context, field_context)
                  VALUES(?,?,?,?,?,?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, info)
        self.conn.commit()

    # 插入entry_logic表记录
    # info = (uid, logic,logic_name)
    def insert_entry_logic(self, info):
        sql = ''' INSERT INTO entry_logic(uid, logic, logic_name)
                  VALUES(?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, info)
        self.conn.commit()

    # 读取一个新的文件，并将里面的信息分别插入到对应数据库表中，最后将这个读取过的文件移动到annotated_data文件夹
    # 注意：此处直接添加数据，并未进行更多预处理（例如去除对于模型训练而言冗余的数据）
    def read_new_file(self, file_path):
        # file_path = './new_data/503-柏柯羽.xlsx'
        path_items = re.split('[/.-]', file_path)
        tagger_name = path_items[-2]
        uid = str(int(path_items[-3]))
        annotate_time = path_items[-4]

        file_content = pd.read_excel(file_path, dtype=str).dropna(how='all').fillna('')
        sentence_num = file_content.shape[0]

        # 一些初步的数据预处理
        # 将句子id列统一成int型
        file_content['句子id'] = pd.to_numeric(file_content['句子id'], downcast='integer')

        # 更新policy表中的文件状态
        self.update_policy((1, tagger_name, annotate_time, uid))
        # 对于每个句子
        for i in range(sentence_num):
            sid = uid + '_' + str(int(file_content.loc[i, '句子id']))
            # 插入到annotated_sentence表
            sentence = file_content.loc[i, '句子']
            sentence_type = file_content.loc[i, '句子类型']
            self.insert_annotated_sentence((uid, sid, sentence, sentence_type))
            # 插入到entity表
            for j in range(1, 6):
                if file_content.loc[i, '实体' + str(j)] != '' and file_content.loc[i, '实体类别' + str(j)] != '':
                    self.insert_entity((sid, file_content.loc[i, '实体' + str(j)], file_content.loc[i, '实体类别' + str(j)]))
            # 插入到entry表
            for j in range(1, 16):
                if file_content.loc[i, '准入条件id' + str(j)] != '':
                    self.insert_entry((sid, file_content.loc[i, '准入条件id' + str(j)], file_content.loc[i, '准入条件类别' + str(j)],
                                  file_content.loc[i, '变量' + str(j)], file_content.loc[i, '关系' + str(j)],
                                  file_content.loc[i, '域' + str(j)],
                                  file_content.loc[i, '变量{}的原文对应'.format(str(j))],
                                  file_content.loc[i, '域{}的原文对应'.format(str(j))]))
            # 插入到entry_logic表
            if file_content.loc[i, '准入条件逻辑关系'] != '':
                self.insert_entry_logic((uid, file_content.loc[i, '准入条件逻辑关系'], file_content.loc[i, '准入条件认定组别']))

        # 将这个文件移动到annotated_data文件夹
        shutil.move(file_path, './annotated_data/{}'.format(file_path.split('/')[-1]))

        self.conn.commit()

    # 将new_data文件夹中的所有新标注文件在数据库中留下记录，并且最后删除new_data中的空文件夹
    def add_new_data(self, directory_path='./new_data'):
        annotated_folder_list = glob.glob('{}/*-*'.format(directory_path))
        for annotated_folder in annotated_folder_list:
            annotated_files = glob.glob('{}/*.xlsx'.format(annotated_folder))
            for file in annotated_files:
                if '~' in file:
                    continue
                print(file)
                self.read_new_file(file)
            # 将原本的文件夹删除
            os.rmdir(annotated_folder)


class DB2DataSet:
    def __init__(self, db_name='policy.db', dataset_path='./datasets/'):
        self.conn = sqlite3.connect(db_name)
        self.dataset_path = dataset_path

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    # 关闭数据库连接，最好在使用对象的最后调用一下
    def close_db(self):
        self.conn.close()

    # sentence_classification数据集converter
    # TODO：后续可以考虑增加前后上下文句子拼接
    def generate_sentence_classification_dataset(self):
        """
        sentence_classification_dataset:
        json数据
        uid, sid, sentence, sentence_type
        """
        dataset_path = self.dataset_path + 'sentence_classification.json'
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        c = self.conn.cursor()
        c.execute("""select * from annotated_sentence""")
        values = c.fetchall()

        keys = ['uid', 'sid', 'sentence', 'sentence_type']
        data = [dict(zip(keys, value)) for value in values]

        with open(dataset_path, 'w') as f:
            json.dump(data, f)

    # entity数据集converter
    def generate_entity_dataset(self):
        """
        entity_dataset
        json数据
        sid, sentence, entity_list:[(entity, entity_type, entity_span)]
        """
        def exceldate2datetime(excel_stamp):
            delta = pd.Timedelta(str(excel_stamp) + 'D')
            real_time = str(pd.to_datetime('1899-12-30') + delta)
            return real_time

        def date_standarder(origin_time, sentence):
            if origin_time in sentence: return origin_time
            # 是excel五位数字日期戳格式
            if re.match('\d{5}$', origin_time):
                origin_time = exceldate2datetime(origin_time)
            time_match = re.search(r"(\d{4}.{0,1}\d{0,2}.{0,1}\d{0,2})", origin_time)
            if not time_match: return origin_time
            time_nums = re.findall(r"(\d+)", time_match.group(0))
            if len(time_nums) == 1 and len(time_nums[0]) == 4:
                time_pattern = time_nums[0] + "年*"
                standard_time = re.search(time_pattern, sentence)
                if not standard_time: return ''
                return standard_time.group(0)
            if len(time_nums) == 2 and len(time_nums[0]) == 4:
                if len(time_nums[1]) == 2 and time_nums[1][0] == '0':
                    time_nums[1] = time_nums[1][1]
                time_pattern = time_nums[0] + ".0{0,1}" + time_nums[1] + "月*"
                standard_time = re.search(time_pattern, sentence)
                if not standard_time: return ''
                return standard_time.group(0)
            if len(time_nums) == 2 and len(time_nums[0]) <= 2:
                if len(time_nums[0]) == 2 and time_nums[0][0] == '0':
                    time_nums[0] = time_nums[0][1]
                if len(time_nums[1]) == 2 and time_nums[1][0] == '0':
                    time_nums[1] = time_nums[1][1]
                time_pattern = "0{0,1}" + time_nums[0] + ".0{0,1}" + time_nums[1] + "日*号*"
                standard_time = re.search(time_pattern, sentence)
                if not standard_time: return ''
                return standard_time.group(0)
            if len(time_nums) >= 3:
                if len(time_nums[1]) == 2 and time_nums[1][0] == '0':
                    time_nums[1] = time_nums[1][1]
                if len(time_nums[2]) == 2 and time_nums[2][0] == '0':
                    time_nums[2] = time_nums[2][1]
                time_pattern = time_nums[0] + ".0{0,1}" + time_nums[1] + ".0{0,1}" + time_nums[2] + "日*号*"
                standard_time = re.search(time_pattern, sentence)
                if not standard_time: return ''
                return standard_time.group(0)

        dataset_path = self.dataset_path + 'entity.json'
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        c = self.conn.cursor()
        c.execute(
            """select entity.sid,entity.entity,entity.entity_type,annotated_sentence.sentence from entity 
            left outer join annotated_sentence on entity.sid=annotated_sentence.sid""")
        values = c.fetchall()  # sid, entity, entity_type, sentence

        legal_values_in_sentence = collections.defaultdict(list)
        sid2sentence = {}
        for i, value in enumerate(values):
            if value[0] not in sid2sentence:
                sid2sentence[value[0]] = value[3]
            # 发布时间由于使用excel，有比较多需要处理的地方
            if value[2] == '发布时间':
                standard_time = date_standarder(value[1], value[3])
                value = (value[0], standard_time, value[2], value[3])

            if value[1] not in value[3]:
                print('not match:', value)
            else:
                start = value[3].find(value[1])
                end = start + len(value[1]) - 1
                substring = value[3][start:end + 1]
                if substring != value[1]:
                    print('substring not match:', value)
                else:
                    legal_values_in_sentence[value[0]].append((value[1], value[2], (start, end)))

        legal_values = []
        for k, v in legal_values_in_sentence.items():
            legal_values.append((k, sid2sentence[k], v))
        keys = ['sid', 'sentence', 'entity_list']
        data = [dict(zip(keys, value)) for value in legal_values]

        with open(dataset_path, 'w') as f:
            json.dump(data, f)

    # entry数据集converter
    # TODO：接着写
    def generate_entry_dataset(self):
        """
        entry_dataset
        json数据
        sid, sentence, entry_list
        """
        dataset_path = self.dataset_path + 'entry.json'
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        c = self.conn.cursor()
        c.execute("""select entry.*,annotated_sentence.sentence from entry 
            left outer join annotated_sentence on entry.sid=annotated_sentence.sid""")
        values = c.fetchall()  # sid, eid, entry_type, var, relation, field, var_context, field_context, sentence

        print('test')

    # all converter pipeline
    def generate_all_datasets(self):
        self.generate_sentence_classification_dataset()
        self.generate_entity_dataset()
        self.generate_entry_dataset()


if __name__ == '__main__':
    config = DBConfig()
    if config.add_new_data_to_db:
        db = PolicyDB()

        db.delete_from_table('annotated_sentence')
        db.delete_from_table('entity')
        db.delete_from_table('entry')
        db.delete_from_table('entry_logic')

        db.add_new_data()

        db.close_db()

    if config.convert_db_to_dataset:
        dataset_converter = DB2DataSet()
        # dataset_converter.generate_sentence_classification_dataset()
        dataset_converter.generate_entity_dataset()
        # dataset_converter.generate_entry_dataset()

        # dataset_converter.generate_all_datasets()
        dataset_converter.close_db()
