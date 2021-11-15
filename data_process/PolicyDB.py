import os
import re
import glob
import json
import shutil
import sqlite3
import unicodedata
import collections
import numpy as np
import pandas as pd
# TODO: unicodedata norm


class DBConfig:
    def __init__(self):
        self.add_new_data_to_db = False
        self.convert_db_to_dataset = True
        self.assign_tasks = False
        self.task_week = 'task_1107'
        self.task_num_per_tagger = 20


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

    # 选取policy表中某个状态的所有记录
    def select_status_from_policy(self, status):
        c = self.conn.cursor()
        c.execute("""select * from policy where annotate_status={}""".format(status))
        values = c.fetchall()
        keys = ['uid', 'origin_id', 'policy_name', 'annotate_status', 'tagger_name', 'annotate_time']
        data = [dict(zip(keys, value)) for value in values]
        return data

    def update_tagger_info(self):
        self.delete_from_table(table_name='tagger')
        df = pd.read_csv('./util_data/tagger_1107.csv')
        df['status'] = 1
        df.columns = ['name', 'id', 'email', 'status']
        df = df[['name', 'id', 'status']].dropna()
        df['id'] = df['id'].astype(int).astype(str)
        df.to_sql('tagger', self.conn, if_exists='replace', index=False)
        self.conn.commit()

    # 更新标注人员信息
    def update_tagger_info_1(self):
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
        # annotate_status: 0-未标注、1-已标注、-1-冗余文件、-2-不需要标注, 2-正在被某人标注
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
        folder_name, file_name = file_path.split('/')[-2], file_path.split('/')[-1]
        folder_item = folder_name.split('-')
        tagger_name = folder_item[0]
        annotate_time = folder_item[1]
        uid = re.split('[-.]', file_name)[0]
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

    # 删除某个文件的所有记录，并将其policy的annotate_status置为0
    def delete_all_data_of_uid(self, uid):
        # info = (annotate_status, tagger_name, annotate_time, uid)
        policy_info = (0, '', '', uid)
        self.update_policy(policy_info)
        self.delete_from_table(sql="""delete from annotated_sentence where uid='{}'""".format(uid))
        self.delete_from_table(sql="""delete from entity where sid LIKE '{}%'""".format(uid))
        self.delete_from_table(sql="""delete from entry where sid LIKE '{}%'""".format(uid))
        self.delete_from_table(sql="""delete from entry_logic where uid='{}'""".format(uid))

    # 清洗冗余文件，将其在policy表中的annotate_status置为-1
    def clean_duplicate_policy(self):

        c = self.conn.cursor()
        c.execute("""select * from policy""")
        values = c.fetchall()

        keys = ['uid', 'origin_id', 'policy_name', 'annotate_status', 'tagger_name', 'annotate_time']
        data = [dict(zip(keys, value)) for value in values]
        data = pd.DataFrame(data)

        # 保留policy name中的汉字字符
        data['clean_policy_name'] = data['policy_name'].apply(lambda x: re.sub('[^\u4e00-\u9fa5]+', '', x))

        # 查找clean policy name冗余的记录
        duplicated = data[data['clean_policy_name'].duplicated(keep=False)]
        duplicated = duplicated.sort_values(by='clean_policy_name').reset_index(drop=True)

        for i in range(1, duplicated.shape[0]):
            current_status = duplicated.loc[i, 'annotate_status']
            if current_status != 0:
                continue
            if duplicated.loc[i, 'clean_policy_name'] == duplicated.loc[i-1, 'clean_policy_name']:
                # duplicated.loc[i, 'annotate_status'] = -1
                policy_info = (-1, '', '', duplicated.loc[i, 'uid'])
                self.update_policy(policy_info)


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
        c.execute("""select annotated_sentence.*, temp.sentence_num 
        from annotated_sentence left outer join (
        select count(sentence) as sentence_num, uid from annotated_sentence group by uid) as temp 
        on annotated_sentence.uid=temp.uid""")
        values = c.fetchall()

        keys = ['uid', 'sid', 'sentence', 'sentence_type', 'sentence_num']
        data = [dict(zip(keys, value)) for value in values]

        with open(dataset_path, 'w') as f:
            json.dump(data, f)

    # entity数据集converter
    # TODO: 是否要采样一些没有entity的句子加入数据集
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

        def department_standarder(origin_depart: str) -> list:
            if '、' in origin_depart or ',' in origin_depart or '，' in origin_depart or ' ' in origin_depart:
                pause_pos = []
                bracket_flag = False
                pause = ['、', ',', '，', ' ']
                left_bracket = ['(', '（', '[', '{', '【', '「', '<', '《']
                right_bracket = [')', '）', ']', '}', '】', '」', '>', '》']
                for i, x in enumerate(origin_depart):
                    if x in left_bracket:
                        bracket_flag = True
                    elif x in right_bracket:
                        bracket_flag = False
                    elif x in pause and not bracket_flag:
                        pause_pos.append(i)
                pause_pos = [-1] + pause_pos
                parts = [origin_depart[i + 1:j] for i, j in zip(pause_pos, pause_pos[1:]+[None])]
                return parts
            else:
                pause_pos = []
                for i, x in enumerate(origin_depart):
                    if i >= 2 and origin_depart[i - 2: i + 1] == '办公室' and i != len(origin_depart) - 1:
                        pause_pos.append(i + 1)
                    if x == '局' and i + 3 <= len(origin_depart) - 1 and (
                            origin_depart[i + 3] == '市' or origin_depart[i + 3] == '区' or origin_depart[i + 3] == '县'):
                        pause_pos.append(i + 1)
                pause_pos = [0] + pause_pos
                parts = [origin_depart[i:j] for i, j in zip(pause_pos, pause_pos[1:]+[None])]

                if not parts:
                    parts.append(origin_depart)
                return parts

        dataset_path = self.dataset_path + 'entity.json'
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        c = self.conn.cursor()
        c.execute(
            """select entity.sid,entity.entity,entity.entity_type,annotated_sentence.sentence,temp.sentence_num from entity 
            left outer join annotated_sentence on entity.sid=annotated_sentence.sid
            left outer join (select count(sentence) as sentence_num, uid from annotated_sentence group by uid) as temp 
            on annotated_sentence.uid=temp.uid""")
        values = c.fetchall()  # sid, entity, entity_type, sentence

        # 处理制定部门、执行部门的历史遗留问题：一个span里有多个部门，要拆开
        refined_values = []
        for i, value in enumerate(values):
            if value[2] != '制定部门' and value[2] != '执行部门':
                refined_values.append(value)
            else:
                refined = department_standarder(value[1])
                for depart in refined:
                    refined_values.append((value[0], depart, value[2], value[3]))
        values = refined_values

        legal_values_in_sentence = collections.defaultdict(list)
        sid2sentence = {}
        sid2sentence_num = {}
        for i, value in enumerate(values):
            if value[0] not in sid2sentence:
                sid2sentence[value[0]] = value[3]
                sid2sentence_num[value[0]] = value[-1]
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
            legal_values.append((k, sid2sentence[k], sid2sentence_num[k], v))
        keys = ['sid', 'sentence', 'sentence_num', 'entity_list']
        data = [dict(zip(keys, value)) for value in legal_values]

        with open(dataset_path, 'w') as f:
            json.dump(data, f)

    # entry数据集converter
    # TODO: 是否要采样一些没有entry的句子加入数据集
    def generate_entry_dataset(self):
        """
        entry_dataset
        json数据
        sid, sentence,
        entry_list:[(eid, entry_type, var, relation, field, var_context, field_context, var_span, field_span)]
        """
        def find_span(origin, context, sentence):
            if origin in sentence:
                start = sentence.find(origin)
                end = start + len(origin) - 1
                substring = sentence[start:end + 1]
                if substring != origin:
                    print('substring not match:', origin, sentence)
                    return None
                return start, end
            if context == '':
                return -1, -1
            else:
                if context not in sentence:
                    print('context not match:', context, sentence)
                    return None
                start = sentence.find(context)
                end = start + len(context) - 1
                substring = sentence[start:end + 1]
                if substring != context:
                    print('substring not match:', context, sentence)
                    return None
            return start, end

        dataset_path = self.dataset_path + 'entry.json'
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        c = self.conn.cursor()
        c.execute("""select entry.*,annotated_sentence.sentence from entry 
            left outer join annotated_sentence on entry.sid=annotated_sentence.sid""")
        values = c.fetchall()  # sid, eid, entry_type, var, relation, field, var_context, field_context, sentence

        legal_values_in_sentence = collections.defaultdict(list)
        sid2sentence = {}
        for i, value in enumerate(values):
            if value[0] not in sid2sentence:
                sid2sentence[value[0]] = value[-1]
            var_span = find_span(value[3], value[6], value[-1])
            if not var_span: continue
            field_span = find_span(value[5], value[7], value[-1])
            if not field_span: continue
            legal_values_in_sentence[value[0]].append(
                (value[1], value[2], value[3], value[4], value[5], value[6], value[7], var_span, field_span))

        legal_values = []
        for k, v in legal_values_in_sentence.items():
            legal_values.append((k, sid2sentence[k], v))
        keys = ['sid', 'sentence', 'entry_list']
        data = [dict(zip(keys, value)) for value in legal_values]

        with open(dataset_path, 'w') as f:
            json.dump(data, f)

    # all converter pipeline
    def generate_all_datasets(self):
        self.generate_sentence_classification_dataset()
        self.generate_entity_dataset()
        self.generate_entry_dataset()


class Tasks:
    def __init__(self, db_name='policy.db', task_path='./tasks/', week='', task_num_per_tagger = 10):
        self.conn = sqlite3.connect(db_name)
        self.task_path = os.path.join(task_path, week)
        self.task_num_per_tagger = task_num_per_tagger
        self.week = week
        self.db = PolicyDB()

        if not os.path.exists(task_path):
            os.makedirs(task_path)
        if not os.path.exists(self.task_path):
            os.makedirs(self.task_path)

    # 查看目前在工作的标注人员
    def get_tagger(self):
        c = self.conn.cursor()
        c.execute("""select * from tagger where status=1""")
        values = c.fetchall()
        keys = ['name', 'id', 'status']
        tagger = [dict(zip(keys, value)) for value in values]
        return tagger

    # 查看目前没被标注的政策pi
    def get_unannotated_uid(self):
        c = self.conn.cursor()
        c.execute("""select * from policy where annotate_status=0""")
        values = c.fetchall()
        keys = ['uid', 'origin_id', 'policy_name', 'annotate_status', 'tagger_name', 'annotate_time']
        policies = [dict(zip(keys, value)) for value in values]
        return policies

    def generate_tasks(self):
        available_taggers = self.get_tagger()
        available_policies = self.get_unannotated_uid()
        total_policy_num = min(len(available_taggers) * self.task_num_per_tagger, len(available_policies))

        tagger_policy_dict = collections.defaultdict(list)
        policy_index = 0
        for tagger in available_taggers:
            count = 0
            while policy_index < total_policy_num and count < 20:
                policy_uid = available_policies[policy_index]['uid']
                tagger_policy_dict[tagger['name']].append(policy_uid)
                policy_index += 1
                count += 1

        for tagger, uid_list in tagger_policy_dict.items():
            dir_name = os.path.join(self.task_path, '{}-{}'.format(tagger, self.week))
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            for uid in uid_list:
                shutil.copytree('/Volumes/TOURO Mobil/policy标注/政策文件rawdata/{}'.format(uid), dir_name+'/{}'.format(uid))
                policy_info = (2, tagger, '', uid)
                self.db.update_policy(policy_info)


if __name__ == '__main__':
    config = DBConfig()

    if config.add_new_data_to_db:
        db = PolicyDB()

        # db.clean_duplicate_policy()

        db.delete_all_data_of_uid('16-')
        db.delete_all_data_of_uid('17-')
        db.delete_all_data_of_uid('18-')
        db.delete_all_data_of_uid('19-')

        # db.delete_from_table('annotated_sentence', )
        # db.delete_from_table('entity')
        # db.delete_from_table('entry')
        # db.delete_from_table('entry_logic')

        db.add_new_data()
        db.close_db()

    if config.convert_db_to_dataset:
        dataset_converter = DB2DataSet()
        # dataset_converter.generate_sentence_classification_dataset()
        # dataset_converter.generate_entity_dataset()
        dataset_converter.generate_entry_dataset()

        # dataset_converter.generate_all_datasets()
        dataset_converter.close_db()

    if config.assign_tasks:
        # db = PolicyDB()
        # db.update_tagger_info()

        # db = PolicyDB()
        # policies = db.select_status_from_policy(2)
        # for policy in policies:
        #     info = (0, '', '', policy['uid'])
        #     db.update_policy(info)

        task_assigner = Tasks(week=config.task_week, task_num_per_tagger=config.task_num_per_tagger)
        task_assigner.generate_tasks()
