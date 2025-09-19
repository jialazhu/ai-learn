import ssl
import certifi
import urllib.request

order_data = [
    {
        "id": 1,
        "text": "我的华为Mate 60订单ORD876543昨天发货，什么时候能到上海？",
        "entities": [
            {"text": "华为Mate 60", "label": "PRODUCT", "start": 2, "end": 12},
            {"text": "ORD876543", "label": "ORDER", "start": 14, "end": 23},
            {"text": "昨天", "label": "TIME", "start": 24, "end": 26},
            {"text": "上海", "label": "LOCATION", "start": 33, "end": 35}
        ]
    },
    {
        "id": 2,
        "text": "2024年10月购买的iPhone 15 Pro价格是8999元，订单号ORD123789能退货吗？",
        "entities": [
            {"text": "iPhone 15 Pro", "label": "PRODUCT", "start": 10, "end": 22},
            {"text": "ORD123789", "label": "ORDER", "start": 32, "end": 41},
            {"text": "2024年10月", "label": "TIME", "start": 0, "end": 8},
            {"text": "8999元", "label": "PRICE", "start": 23, "end": 28}
        ]
    },
    {
        "id": 3,
        "text": "深圳仓发货的小米14，订单ORD456123，价格￥4299，预计明天送达吗？",
        "entities": [
            {"text": "小米14", "label": "PRODUCT", "start": 6, "end": 11},
            {"text": "ORD456123", "label": "ORDER", "start": 14, "end": 23},
            {"text": "明天", "label": "TIME", "start": 30, "end": 32},
            {"text": "深圳", "label": "LOCATION", "start": 0, "end": 2},
            {"text": "￥4299", "label": "PRICE", "start": 24, "end": 29}
        ]
    },
    {
        "id": 4,
        "text": "上个月买的OPPO Find X7 Ultra，订单ORD901324，在广州能换货吗？",
        "entities": [
            {"text": "OPPO Find X7 Ultra", "label": "PRODUCT", "start": 4, "end": 21},
            {"text": "ORD901324", "label": "ORDER", "start": 24, "end": 33},
            {"text": "上个月", "label": "TIME", "start": 0, "end": 3},
            {"text": "广州", "label": "LOCATION", "start": 34, "end": 36}
        ]
    },
    {
        "id": 5,
        "text": "订单ORD234567的vivo X100，2024年9月15日下单，现在想改地址到武汉",
        "entities": [
            {"text": "vivo X100", "label": "PRODUCT", "start": 14, "end": 22},
            {"text": "ORD234567", "label": "ORDER", "start": 3, "end": 12},
            {"text": "2024年9月15日", "label": "TIME", "start": 23, "end": 34},
            {"text": "武汉", "label": "LOCATION", "start": 46, "end": 48}
        ]
    },
    {
        "id": 6,
        "text": "3天前收到的荣耀Magic V2折叠屏，价格5999元，订单ORD567890有质量问题",
        "entities": [
            {"text": "荣耀Magic V2折叠屏", "label": "PRODUCT", "start": 6, "end": 19},
            {"text": "ORD567890", "label": "ORDER", "start": 30, "end": 39},
            {"text": "3天前", "label": "TIME", "start": 0, "end": 3},
            {"text": "5999元", "label": "PRICE", "start": 22, "end": 26}
        ]
    },
    {
        "id": 7,
        "text": "上海用户问，订单ORD678901的三星Galaxy S24 Ultra下周能送到吗？",
        "entities": [
            {"text": "三星Galaxy S24 Ultra", "label": "PRODUCT", "start": 20, "end": 38},
            {"text": "ORD678901", "label": "ORDER", "start": 11, "end": 20},
            {"text": "下周", "label": "TIME", "start": 39, "end": 41},
            {"text": "上海", "label": "LOCATION", "start": 0, "end": 2}
        ]
    },
    {
        "id": 8,
        "text": "2024年双11购买的iPad Air 6，订单ORD789012，价格￥5599，怎么查询物流？",
        "entities": [
            {"text": "iPad Air 6", "label": "PRODUCT", "start": 10, "end": 19},
            {"text": "ORD789012", "label": "ORDER", "start": 22, "end": 31},
            {"text": "2024年双11", "label": "TIME", "start": 0, "end": 8},
            {"text": "￥5599", "label": "PRICE", "start": 32, "end": 37}
        ]
    },
    {
        "id": 9,
        "text": "我的订单ORD890123买的是一加12，昨天付款的，能加急送到成都吗？",
        "entities": [
            {"text": "一加12", "label": "PRODUCT", "start": 17, "end": 22},
            {"text": "ORD890123", "label": "ORDER", "start": 4, "end": 13},
            {"text": "昨天", "label": "TIME", "start": 23, "end": 25},
            {"text": "成都", "label": "LOCATION", "start": 35, "end": 37}
        ]
    },
    {
        "id": 10,
        "text": "真我GT7 Pro订单ORD901234，2024年8月购买价格3299元，北京售后在哪？",
        "entities": [
            {"text": "真我GT7 Pro", "label": "PRODUCT", "start": 0, "end": 9},
            {"text": "ORD901234", "label": "ORDER", "start": 11, "end": 20},
            {"text": "2024年8月", "label": "TIME", "start": 21, "end": 28},
            {"text": "北京", "label": "LOCATION", "start": 38, "end": 40},
            {"text": "3299元", "label": "PRICE", "start": 32, "end": 36}
        ]
    }
]


def correct_entity_positions(data):
    """
    修正实体的start和end位置
    :param data: 原始订单数据（包含id、text、entities）
    :return: 修正后的订单数据
    """
    corrected_data = []
    for item in data:
        sentence = item["text"]
        entities = item["entities"]
        corrected_entities = []

        print(f"\n=== 处理ID为 {item['id']} 的句子 ===")
        print(f"原始句子：{sentence}")

        for entity in entities:
            entity_text = entity["text"]
            old_start = entity["start"]
            old_end = entity["end"]

            # 1. 找到实体在句子中的真实起始位置
            real_start = sentence.find(entity_text)
            # 2. 计算真实结束位置（start + 实体长度）
            real_end = real_start + len(entity_text)

            # 3. 存储修正后的实体信息
            corrected_entity = {
                "text": entity_text,
                "label": entity["label"],
                "start": real_start,
                "end": real_end,
                "old_start": old_start,  # 保留原始值用于对比
                "old_end": old_end  # 保留原始值用于对比
            }
            corrected_entities.append(corrected_entity)

            # 打印修正对比信息
            print(f"- 实体「{entity_text}」：原始({old_start},{old_end}) → 修正后({real_start},{real_end})")

        # 4. 组装修正后的item
        corrected_item = {
            "id": item["id"],
            "text": sentence,
            "entities": [
                # 移除临时的old_start/old_end，只保留最终的实体信息
                {"text": e["text"], "label": e["label"], "start": e["start"], "end": e["end"]}
                for e in corrected_entities
            ]
        }
        corrected_data.append(corrected_item)

    return corrected_data


# 执行修正
corrected_order_data = correct_entity_positions(order_data)

# 可选：打印最终修正结果（完整数据）
print("\n" + "=" * 50)
print("最终修正后的数据（可直接复制使用）：")
print(corrected_order_data)
