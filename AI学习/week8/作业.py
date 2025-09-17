import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from TorchCRF import CRF


json = {
  "questions": [
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
        {"text": "荣耀Magic V2折叠屏", "label": "PRODUCT", "start": 5, "end": 21},
        {"text": "ORD567890", "label": "ORDER", "start": 31, "end": 40},
        {"text": "3天前", "label": "TIME", "start": 0, "end": 3},
        {"text": "5999元", "label": "PRICE", "start": 22, "end": 27}
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
        {"text": "真我GT7 Pro", "label": "PRODUCT", "start": 0, "end": 11},
        {"text": "ORD901234", "label": "ORDER", "start": 12, "end": 21},
        {"text": "2024年8月", "label": "TIME", "start": 22, "end": 30},
        {"text": "北京", "label": "LOCATION", "start": 39, "end": 41},
        {"text": "3299元", "label": "PRICE", "start": 31, "end": 36}
      ]
    }
  ],
  "faq": [
    {
      "id": 1,
      "question": "如何查询订单物流信息？",
      "answer": "您可以在APP首页点击「我的订单」，选择对应订单即可查看实时物流信息，也可通过订单详情页的「追踪物流」按钮查询。"
    },
    {
      "id": 2,
      "question": "订单发货后多久能送达？",
      "answer": "一般情况下，发货后同城1-2天，省内2-3天，跨省3-5天送达，偏远地区可能延迟1-2天。"
    },
    {
      "id": 3,
      "question": "可以修改订单的收货地址吗？",
      "answer": "订单未发货前可在「我的订单」中申请修改地址；已发货订单请联系客服协助处理，可能产生改派费用。"
    },
    {
      "id": 4,
      "question": "产品支持7天无理由退货吗？",
      "answer": "支持，在产品完好、不影响二次销售且购买不超过7天的情况下，可申请无理由退货，部分特殊商品除外。"
    },
    {
      "id": 5,
      "question": "如何申请产品换货？",
      "answer": "请在「我的订单」中选择对应订单，点击「申请换货」，填写换货原因并上传相关凭证，审核通过后按指引操作即可。"
    },
    {
      "id": 6,
      "question": "订单取消后退款多久到账？",
      "answer": "取消订单后，退款将在1-3个工作日原路返回，具体到账时间以银行处理速度为准。"
    },
    {
      "id": 7,
      "question": "可以加急配送吗？",
      "answer": "部分城市支持加急配送服务，您可在下单时选择「加急配送」选项，将额外收取20元加急费，具体以页面显示为准。"
    },
    {
      "id": 8,
      "question": "产品保修期是多久？",
      "answer": "不同产品保修期不同，手机、电脑等数码产品通常为1年，配件类产品通常为3-6个月，具体以产品说明书为准。"
    },
    {
      "id": 9,
      "question": "如何开具发票？",
      "answer": "您可以在下单时选择需要发票，或在收货后30天内到「我的订单」中申请补开发票，电子发票将发送至您的邮箱。"
    },
    {
      "id": 10,
      "question": "订单显示已签收但未收到货怎么办？",
      "answer": "请先检查是否为家人或小区物业代收，如确认未收到，请立即联系客服并提供订单号，我们将协助核实处理。"
    },
    {
      "id": 11,
      "question": "支持货到付款吗？",
      "answer": "部分商品和城市支持货到付款服务，您可在结算页面查看是否有「货到付款」选项，支持现金和POS机刷卡。"
    },
    {
      "id": 12,
      "question": "产品有质量问题如何保修？",
      "answer": "请携带产品、购买凭证和保修卡到官方授权服务中心，或通过APP提交保修申请，我们将根据检测结果提供保修服务。"
    },
    {
      "id": 13,
      "question": "可以指定送货时间吗？",
      "answer": "支持，下单时可选择「预约配送」，设置具体送达日期和时间段（9:00-12:00、14:00-18:00、19:00-21:00）。"
    },
    {
      "id": 14,
      "question": "如何修改订单的付款方式？",
      "answer": "未付款订单可直接取消后重新下单选择其他付款方式；已付款订单无法修改付款方式，如有需要请联系客服。"
    },
    {
      "id": 15,
      "question": "购买的产品可以退换颜色吗？",
      "answer": "在产品未拆封使用且不影响二次销售的情况下，可在收货后7天内申请更换颜色，需自行承担来回运费。"
    },
    {
      "id": 16,
      "question": "订单超过多久未付款会自动取消？",
      "answer": "普通商品订单未付款将在24小时后自动取消，活动期间部分商品订单未付款将在15分钟后自动取消。"
    },
    {
      "id": 17,
      "question": "支持跨城市退货吗？",
      "answer": "支持，您可以通过APP申请退货后，将商品寄回指定仓库，退款将在收到退回商品并检测合格后3个工作日内到账。"
    },
    {
      "id": 18,
      "question": "如何查询附近的线下服务中心？",
      "answer": "在APP首页点击「服务中心」，允许获取位置信息后即可查看附近的线下服务中心地址、营业时间和联系方式。"
    },
    {
      "id": 19,
      "question": "购买后发现产品降价了可以补差价吗？",
      "answer": "在收货后7天内，如遇产品降价，可申请价格保护并退还差价，具体以活动规则为准，部分特殊活动不支持。"
    },
    {
      "id": 20,
      "question": "可以更换订单的配送方式吗？",
      "answer": "未发货订单可联系客服申请更换配送方式；已发货订单无法更换配送方式，如有特殊需求请联系客服协助。"
    }
  ]
}


