"""MySQL 连接、checkpoint 初始化和本地单用户会话目录。"""

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pymysql
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql.pymysql import PyMySQLStore


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str = "jobagent"

    @classmethod
    def from_env(cls):
        user = os.getenv("MYSQL_USER", "")
        if not user or "MYSQL_PASSWORD" not in os.environ:
            raise ValueError("请在 .env 配置 MYSQL_USER 和 MYSQL_PASSWORD，再运行 python -m scripts.init_mysql。")
        database = os.getenv("MYSQL_DATABASE", "jobagent")
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", database):
            raise ValueError("MYSQL_DATABASE 只能包含字母、数字和下划线。")
        return cls(os.getenv("MYSQL_HOST", "localhost"), int(os.getenv("MYSQL_PORT", "3306")),
                   user, os.environ["MYSQL_PASSWORD"], database)

    def connect(self, *, with_database=True):
        # 分字段传递，密码中的 @、: 等字符无需 URL 编码。
        return pymysql.connect(
            host=self.host, port=self.port, user=self.user, password=self.password,
            database=self.database if with_database else None,
            charset="utf8mb4", autocommit=True, connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )


class ThreadRepository:
    """会话列表只存标题和时间；消息和图执行位置由 checkpointer 管理。"""

    def __init__(self, connection):
        self.connection = connection

    def list_threads(self):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT thread_id, title, updated_at FROM app_threads ORDER BY updated_at DESC, thread_id")
            return list(cursor.fetchall())

    def create(self, title="新会话"):
        thread_id = str(uuid4())
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO app_threads (thread_id, title) VALUES (%s, %s)", (thread_id, title[:120]))
        return thread_id

    def touch(self, thread_id, title=None):
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE app_threads SET title = IF(title = '新会话' AND %s IS NOT NULL, %s, title), "
                "updated_at = CURRENT_TIMESTAMP(6) WHERE thread_id = %s",
                (title[:120] if title else None, title[:120] if title else None, thread_id),
            )

    @contextmanager
    def lock(self, thread_id):
        # 同一会话跨页面/进程也只能有一个执行者，避免完整历史互相覆盖。
        name = f"jobagent:{thread_id}"
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, 0) AS acquired", (name,))
            if cursor.fetchone()["acquired"] != 1:
                raise RuntimeError("该会话正在另一个页面执行，请稍后重试。")
        try:
            yield
        finally:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT RELEASE_LOCK(%s)", (name,))


@dataclass
class DatabaseSession:
    checkpointer: PyMySQLSaver
    threads: ThreadRepository
    store: PyMySQLStore


@contextmanager
def open_database(settings=None):
    """每次页面执行独立连接；异常、停止或 rerun 后都关闭，不缓存失效连接。"""
    settings = settings or MySQLSettings.from_env()
    # Store 与 saver 的锁彼此独立，因此不能共用可能被后台保存任务使用的连接。
    with settings.connect() as connection, settings.connect() as store_connection:
        yield DatabaseSession(PyMySQLSaver(connection), ThreadRepository(connection), PyMySQLStore(store_connection))


def initialize_database(settings=None):
    settings = settings or MySQLSettings.from_env()
    with settings.connect(with_database=False) as connection:
        with connection.cursor() as cursor:
            # 适配器的 JSON_TABLE 使用 MySQL 8 默认 utf8mb4 排序规则，表需与其一致。
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{settings.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
    with settings.connect() as connection:
        PyMySQLSaver(connection).setup()
        PyMySQLStore(connection).setup()
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_threads (
                    thread_id CHAR(36) PRIMARY KEY,
                    title VARCHAR(120) NOT NULL,
                    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                    INDEX idx_threads_updated (updated_at)
                ) ENGINE=InnoDB
            """)
    return settings.database
