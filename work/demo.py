import datetime


def generate_insert_sql(
        start_date: str,  # 开始日期，格式：YYYY-MM-DD
        end_date: str,  # 结束日期，格式：YYYY-MM-DD
        numbers: list,  # 要生成的编号列表（对应示例中的第三个字段）
        time_slots: dict,  # 时间段映射，key=时间段字符串，value=最后一位数字
        fixed_values: tuple,  # 固定字段值 (col1, col2, col7, col8)
        table_name: str = "your_table_name"  # 目标表名
) -> str:
    """
    生成批量插入SQL语句
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param numbers: 编号列表，如 [1,2,3]
    :param time_slots: 时间段配置，如 {'11:00-14:00': 1, '16:00-21:00': 2}
    :param fixed_values: 固定字段值 (col1, col2, col7, col8)
    :param table_name: 表名
    :return: 完整的INSERT SQL语句
    """
    # 解析日期
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # 字段名（可根据实际表结构修改）
    columns = "(sub_service_id, doctor_id, o2o_address_id, date, period_name, max_num, available_num, sort)"

    # 生成VALUES部分
    values_lines = []
    current_date = start

    # 遍历每一天
    while current_date <= end:
        # 跳过周六和周日
        # if current_date.weekday() >= 6:
        #     continue
        date_str = current_date.strftime("%Y-%m-%d")

        # 遍历每个编号
        for num in numbers:
            # 遍历每个时间段
            for idx, (time_slot, last_num) in enumerate(time_slots.items()):
                # 拼接一行数据
                line = f"({fixed_values[0]}, {fixed_values[1]}, {num}, '{date_str}', '{time_slot}', {last_num}, {last_num}, {idx+1})"
                values_lines.append(line)

        # 日期+1天
        current_date += datetime.timedelta(days=1)

    # 拼接完整SQL
    values_str = ",\n".join(values_lines)
    sql = f"INSERT INTO {table_name} {columns}\nVALUES\n{values_str};"

    return sql


# ====================== 配置参数（根据需要修改）======================
if __name__ == "__main__":
    # 基础配置
    CONFIG = {
        "start_date": "2026-03-23",  # 开始日期
        "end_date": "2026-04-03",  # 结束日期
        "numbers": [1, 2, 3],  # 要生成的编号（第三个字段）地址id
        # 时间段配置：键=时间段字符串，值=库存
        "time_slots": {
            "11:00-14:00": 5,
            "14:00-16:00": 5,
            "16:00-18:00": 5
        },
        # 固定字段值 (col1 子服务id, col2 护士id)
        "fixed_values": (1, 1),
        "table_name": "tb_channel_o2o_subservice_capacity"  # 替换为实际表名
    }

    # 生成SQL
    sql = generate_insert_sql(**CONFIG)

    # 输出SQL（可选：写入文件）
    print("生成的SQL语句：")
    print("=" * 50)
    print(sql)

    # 可选：将SQL写入文件
    # with open("batch_insert.sql", "w", encoding="utf-8") as f:
    #     f.write(sql)
    # print("\nSQL已写入 batch_insert.sql 文件")