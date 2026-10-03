"""显式创建项目数据库和表，不打印数据库密码。"""

from dotenv import load_dotenv
from persistence import initialize_database
from settings import PROJECT_ROOT


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        database = initialize_database()
    except Exception as exc:
        # 不显示连接对象、URL 或完整异常参数。
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else type(exc).__name__
        raise SystemExit(f"MySQL 初始化失败（{code}），请检查 .env、服务状态与建库权限。") from None
    print(f"MySQL 数据库 {database} 初始化完成：checkpoint 表、app_threads、store 及 store_migrations。")


if __name__ == "__main__":
    main()
