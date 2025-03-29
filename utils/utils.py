import os
import time
import re
import json
import sqlite3
import random
import base64
from copy import deepcopy
from typing import List, Tuple
from colorama import Fore, Back, Style
import streamlit as st
from configs import LOGIN_BACKGROUD_IAMGE, BACKGROUD_IAMGE


def print_colorful(
    *text,
    text_color=None,
    time_color=Back.GREEN + Fore.RED,
    sep: str = " ",
    end: str = "\n",
    file=None,
    flush: bool = False,
):
    timestamp = time.strftime("%y/%m/%d %H:%M:%S") + " : "
    text = sep.join(list(map(str, text)))
    text = text_color + text + Style.RESET_ALL if text_color is not None else text
    print(
        f"{time_color + timestamp + Style.RESET_ALL}{text}",
        end=end,
        file=file,
        flush=flush,
    )


def short_name(name, l=10):
    if len(name) > l:
        name = name[: l // 2] + "..." + name[-l // 2 :]
    return name


class SQLClient:
    def __init__(
        self, path="database/users.db", isolation_level=None, timeout=5
    ) -> None:
        self.conn = sqlite3.connect(
            path, isolation_level=isolation_level, timeout=timeout
        )

    def close(self):
        self.conn.close()

    def update_item(self, table, citem, cvalue, t_item, t_value):
        # 修改某个字段的值
        cvalue = cvalue if isinstance(cvalue, (float, int)) else f"'{cvalue}'"
        t_value = cvalue if isinstance(t_value, (float, int)) else f"'{t_value}'"

        c = self.conn.cursor()
        c.execute(f"UPDATE {table} SET {t_item}={t_value} WHERE {citem}={cvalue}")
        self.conn.commit()

    def insert_item(self, table: str, values):
        # 添加新item
        c = self.conn.cursor()
        values = ", ".join(
            map(lambda x: x if isinstance(x, (int, float)) else f"'{x}'", values)
        )
        sql = f"INSERT INTO {table} VALUES ({values})"
        c.execute(sql)
        self.conn.commit()

    def fetch_all(self, table: str, conditions: str = ""):
        # 获取表table中所有的字段
        c = self.conn.cursor()
        sql = f"SELECT * FROM {table}" + (f" WHERE {conditions}" if conditions else "")
        data = [row for row in c.execute(sql)]
        titles = self.get_col_names(table)

        return data, titles

    def get_table_names(self):
        # 获取表名，保存在table_name列表
        c = self.conn.cursor()
        c.execute("select name from sqlite_master where type='table'")
        rows = c.fetchall()
        table_name = [row[0] for row in rows]
        return table_name

    def get_col_names(self, table):
        """获取table的所有列名"""
        c = self.conn.cursor()
        c.execute(f"pragma table_info({table})")
        col_names = c.fetchall()
        col_names = [x[1] for x in col_names]

        return col_names

    def create_table(self, table: str, items: List[Tuple[str, str]]):
        """创建一个table
        table：表名
        items：项
        """
        c = self.conn.cursor()
        items_str = ", ".join([" ".join(i) for i in items])
        c.execute(f"""CREATE TABLE IF NOT EXISTS {table} ({items_str})""")
        self.conn.commit()

    def delete_item(self, table, item, value):
        """删除表table中，字段item中值为value的所有项"""
        if table in self.get_table_names():
            c = self.conn.cursor()
            c.execute(f"DELETE FROM {table} WHERE {item}='{value}'")
            self.conn.commit()

    def delete_table(self, table):
        """清空表, 不可恢复"""
        if table in self.get_table_names():
            c = self.conn.cursor()
            c.execute(f"DELETE FROM {table};")
            self.conn.commit()

    def __del__(self):
        self.close()


def random_icon(idx=None):
    icons = "🍇🍈🍉🍊🍋🍌🍍🥭🍎🍏🍐🍑🍒🍓"  # 🐭🐁🐀🐹🐰
    n = len(icons)
    if idx is None:
        return random.sample(icons, 1)[0]
    else:
        return icons[idx % n]


@st.cache_resource
def read_user_info(path="database/users.db", cache_path="database/cache_users.json"):
    client = SQLClient(path)
    client.create_table(
        "user_infos",
        [
            ("name", "TEXT"),
            ("email", "TEXT"),
            ("password", "TEXT"),
            ("register_time", "TEXT"),
            ("login_time", "TEXT"),
            ("last_login_time", "TEXT"),
        ],
    )
    data, _ = client.fetch_all(table="user_infos")
    client.close()

    info = {
        d[0]: {
            "email": d[1],
            "password": d[2],
            "register_time": float(d[3]),
            "login_time": float(d[4]),
            "last_login_time": float(d[5]),
        }
        for d in data
    }

    # email 作为key
    info_email = {}
    if info:
        info_email = {
            v.pop("email"): {"name": k, **v} for k, v in deepcopy(info).items()
        }

    # 读取当前缓存数据
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            info.update(json.load(f))

    return info, info_email


def save_user_info(
    info: dict, db_path="database/users.db", cache_path="database/cache_users.json"
):
    # 写入数据库
    client = SQLClient(db_path)

    # 清空
    client.delete_table("user_infos")

    # 保存到数据库
    data = [
        [
            k,
            v.get("email", ""),
            v.get("password", ""),
            str(v.get("register_time", "")),
            str(v.get("login_time", "")),
            str(v.get("last_login_time", "")),
        ]
        for k, v in info.items()
        if k not in ["last_user", "current_user"]
    ]
    for d in data:
        client.insert_item("user_infos", d)
    client.close()

    # 保存到缓存
    with open(cache_path, "w", encoding="utf-8") as f:
        data = {}
        if last_user := info.get("last_user"):
            data["last_user"] = last_user
        if current_user := info.get("current_user"):
            data["current_user"] = current_user

        json.dump(data, f, ensure_ascii=False)


def theme(
    loc="body", show_bg=True, body_padding=(3, 0, 2, 0), siderbar_padding=(3, 1.5)
):
    body_padding = " ".join([f"{i}rem" for i in body_padding])
    siderbar_padding = " ".join([f"{i}rem" for i in siderbar_padding])
    st.markdown(
        """<style>
        .stDeployButton {
                    visibility: hidden;
                }
        .block-container {
            padding: {{body_padding}};
        }
        .st-emotion-cache-16txtl3 {
            padding: {{siderbar_padding}};
        }
        </style>""".replace(
            "{{body_padding}}", body_padding
        ).replace(
            "{{siderbar_padding}}", siderbar_padding
        ),
        unsafe_allow_html=True,
    )

    # 显示背景图片
    main_bg = ""
    if loc == "body" and os.path.exists(BACKGROUD_IAMGE):
        main_bg = BACKGROUD_IAMGE
        main_bg_ext = main_bg.split(".")[-1]
    elif os.path.exists(LOGIN_BACKGROUD_IAMGE):
        main_bg = LOGIN_BACKGROUD_IAMGE
        main_bg_ext = main_bg.split(".")[-1]

    if main_bg and show_bg:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def validate_email(email):
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    if re.match(email_regex, email):
        return True
    else:
        return False
