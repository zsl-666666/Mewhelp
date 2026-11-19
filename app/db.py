"""SQLAlchemy 2.x 数据库连接配置。

只提供 Engine / Session 基础设施，不在此阶段创建任何业务表。
连接参数从项目根目录的 .env 读取。
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import Session, sessionmaker


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "customer_service"
    mysql_user: str = "customer_service"
    mysql_password: SecretStr

    @property
    def url(self) -> URL:
        return URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password.get_secret_value(),
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": "utf8mb4"},
        )


@lru_cache
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


def _set_utf8mb4_collation(dbapi_connection, connection_record) -> None:
    """修正连接排序规则。

    SQLAlchemy 的 MySQL 方言在连接建立后会执行 ``SET NAMES <charset>``，
    这会把 PyMySQL 的 collation 参数覆盖为服务器默认排序规则。
    """
    with dbapi_connection.cursor() as cursor:
        cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci")


@lru_cache
def get_engine() -> Engine:
    engine = create_engine(
        get_database_settings().url,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    event.listen(engine, "connect", _set_utf8mb4_collation)
    return engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db_session_factory() -> sessionmaker[Session]:
    """FastAPI 依赖：返回缓存好的 Session 工厂，供需要自建 session 的服务使用。"""
    return get_session_factory()


def get_db_session() -> Iterator[Session]:
    """FastAPI 依赖：每个请求使用一个 Session，请求结束后关闭。"""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
