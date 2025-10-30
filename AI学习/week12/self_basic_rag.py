import re
from typing import List, Tuple

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RagDemo:

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 4)
        )
        self.is_fitted = False

    def pre_text(self,text:str) -> str:
        text = ' '.join(text.split())

        word = jieba.cut(text)
        return ' '.join(word)

    def chunk_document(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if end < len(text) and text[end] not in ['。', '！', '？', '\n']:
                last_period = chunk.rfind('。')
                if last_period > chunk_size // 2:  # 确保块不会太小
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - overlap
            if start >= len(text):
                break

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]


    def _create_embeddings(self):
        """创建文档嵌入"""
        self.embeddings = self.vectorizer.fit_transform(self.documents).toarray()
        self.is_fitted = True
        print(f"✅ 生成了 {self.embeddings.shape[0]} 个向量，维度为 {self.embeddings.shape[1]}")

    def add_documents(self, documents: List[str]):

        all_chunks = []
        for i, doc in enumerate(documents):
            processed_doc = self.pre_text( doc)
            # chunks = self.chunk_document(processed_doc)
            all_chunks.extend([processed_doc])

        self.documents = all_chunks
        self._create_embeddings()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("请先添加文档并生成嵌入向量")

        pre_query = self.pre_text(query)

        query_vector = self.vectorizer.transform([pre_query])

        simiarities = cosine_similarity(query_vector,self.embeddings)[0]

        top_indices = np.argsort(-simiarities)[::1][:k]

        result = []

        for i, idx in enumerate(top_indices):
            doc = self.documents[idx]
            score = simiarities[idx]
            result.append((doc,score))
            # print(f"  {i + 1}. 相似度: {score:.4f}")
            # print(f"     内容: {doc[:100]}...")

        return result

    def generate_answer(self, query:str,context_docs:List[str]) -> str:

        print("正在生成答案---")
        context = "".join( [f"{doc}" for i , doc in enumerate(context_docs)])
        return context

    def rag_pipeline(self,query:str,k:int=3) -> str:

        context_result = self.search(query,k)

        context_docs = [doc for doc, score in context_result if score > 0.1]

        if not context_docs:
            return "抱歉，没有找到相关的信息来回答您的问题。"

        answer = self.generate_answer(query,context_docs)

        return answer


def create_sample_documents() -> List[str]:
    """创建演示用的样本文档"""
    documents = [
        "血常规是临床最基础的血液检验项目，通过检测血液中红细胞、白细胞、血小板、血红蛋白等指标的数量及形态变化，辅助判断是否存在感染、贫血、血液系统疾病等情况，广泛应用于常规体检、疾病诊断与治疗监测。",
        "靶向治疗是一种精准抗癌治疗方式，依据肿瘤细胞特有的基因变异、蛋白表达等 靶点，使用针对性药物（如小分子靶向药、单克隆抗体）特异性作用于靶点，阻断肿瘤细胞生长信号或破坏肿瘤细胞，对正常细胞损伤较小，常见于肺癌、乳腺癌、白血病等恶性肿瘤治疗。",
        "高血压是以体循环动脉血压持续升高为主要特征的慢性疾病，诊断标准为在未使用降压药的情况下，非同日三次测量收缩压≥140mmHg 和（或）舒张压≥90mmHg。长期高血压会损伤心、脑、肾、眼等靶器官，诱发脑卒中、心肌梗死、肾衰竭等并发症，需通过生活方式干预（如低盐饮食、运动）或药物长期控制。",
        "静脉输液是将药物、营养液、血液制品等通过静脉穿刺输入人体血液循环的治疗手段，能使药物快速起效，适用于病情紧急（如严重感染、休克）、口服药物吸收差或无法口服（如昏迷、呕吐）的患者。但该方式存在感染、静脉炎、药物不良反应等风险，需严格遵循无菌操作规范。",
        "CT 检查即计算机断层扫描，利用 X 线束对人体特定部位进行连续断层扫描，结合计算机处理重建出人体断层图像，可清晰显示器官结构、病变位置及性质，常用于头部外伤、肺部疾病（如肺炎、肺癌）、肿瘤排查、骨骼损伤等诊断，不过存在一定电离辐射，需根据病情合理使用。",
        "糖尿病是一组以慢性血糖水平升高为特征的代谢性疾病，主要分为 1 型（胰岛素分泌绝对不足）和 2 型（胰岛素抵抗或分泌相对不足），典型症状为 三多一少（多饮、多食、多尿、体重减轻）。长期高血糖会引发神经病变、肾病、视网膜病变等并发症，治疗需结合饮食控制、运动、血糖监测及药物（如胰岛素、口服降糖药）。",
        "抗生素是一类用于抑制或杀灭细菌的药物，可治疗细菌感染引起的疾病（如肺炎、尿路感染、皮肤感染），但对病毒感染（如感冒、流感）无效。滥用抗生素会导致细菌耐药性，使药物疗效下降甚至失效，临床需根据感染类型、细菌药敏试验结果合理选择与使用。",
        "心电图（ECG）是通过在人体体表放置电极，记录心脏电活动变化的检查方法，能反映心脏节律、心率、心肌供血情况及心脏结构异常（如心肌梗死、心律失常、心肌炎），操作简便、无创，常用于心脏疾病筛查、手术监护、体检等场景，对急性心肌梗死的早期诊断具有重要意义。",
        "脑卒中俗称 中风，因脑部血管突然破裂（出血性脑卒中）或血管阻塞导致血液无法流入大脑（缺血性脑卒中），引起脑组织损伤的急性脑血管疾病。常见症状为突然肢体麻木、言语不清、口角歪斜、意识障碍等，发病后需及时就医（如缺血性脑卒中溶栓治疗），否则易遗留偏瘫、失语等后遗症。",
        "疫苗是用各类病原微生物（如细菌、病毒、立克次体）经过减毒、灭活或基因工程等技术制备的生物制品，接种后可刺激人体免疫系统产生特异性抗体或免疫细胞，当人体再次接触该病原体时，能快速启动免疫反应，预防感染疾病。常见疫苗包括新冠疫苗、乙肝疫苗、脊髓灰质炎疫苗、流感疫苗等，是预防传染病的重要手段。"
    ]

    return documents

def conversion(text: str):
    documents = [
    ("血常规是临床最基础的血液检验项目，通过检测血液中红细胞、白细胞、血小板、血红蛋白等指标的数量及形态变化，辅助判断是否存在感染、贫血、血液系统疾病等情况，广泛应用于常规体检、疾病诊断与治疗监测。", "Complete Blood Count (CBC) is the most basic clinical blood test. By detecting the quantity and morphological changes of indicators such as red blood cells, white blood cells, platelets, and hemoglobin in the blood, it helps determine the presence of conditions like infections, anemia, and hematological diseases. It is widely used in routine physical examinations, disease diagnosis, and treatment monitoring."),
    ("靶向治疗是一种精准抗癌治疗方式，依据肿瘤细胞特有的基因变异、蛋白表达等 靶点，使用针对性药物（如小分子靶向药、单克隆抗体）特异性作用于靶点，阻断肿瘤细胞生长信号或破坏肿瘤细胞，对正常细胞损伤较小，常见于肺癌、乳腺癌、白血病等恶性肿瘤治疗。", "Targeted therapy is a precise anti-cancer treatment method. Based on unique 'targets' of tumor cells such as genetic mutations and protein expression, it uses targeted drugs (e.g., small-molecule targeted drugs, monoclonal antibodies) to act specifically on these targets, blocking the growth signals of tumor cells or destroying them. It causes less damage to normal cells and is commonly used in the treatment of malignant tumors such as lung cancer, breast cancer, and leukemia."),
    ("高血压是以体循环动脉血压持续升高为主要特征的慢性疾病，诊断标准为在未使用降压药的情况下，非同日三次测量收缩压≥140mmHg 和（或）舒张压≥90mmHg。长期高血压会损伤心、脑、肾、眼等靶器官，诱发脑卒中、心肌梗死、肾衰竭等并发症，需通过生活方式干预（如低盐饮食、运动）或药物长期控制。", "Hypertension is a chronic disease characterized by sustained elevation of systemic arterial blood pressure. The diagnostic criterion is that without the use of antihypertensive drugs, the systolic blood pressure is ≥140 mmHg and/or the diastolic blood pressure is ≥90 mmHg in three non-consecutive measurements on different days. Long-term hypertension can damage target organs such as the heart, brain, kidneys, and eyes, and induce complications like stroke, myocardial infarction, and renal failure. It needs to be controlled for a long time through lifestyle interventions (e.g., low-salt diet, exercise) or medications."),
    ("静脉输液是将药物、营养液、血液制品等通过静脉穿刺输入人体血液循环的治疗手段，能使药物快速起效，适用于病情紧急（如严重感染、休克）、口服药物吸收差或无法口服（如昏迷、呕吐）的患者。但该方式存在感染、静脉炎、药物不良反应等风险，需严格遵循无菌操作规范。", "Intravenous infusion is a treatment method that delivers drugs, nutrient solutions, blood products, etc., into the human bloodstream through venous puncture. It allows drugs to take effect quickly and is suitable for patients in emergency situations (e.g., severe infection, shock), patients with poor absorption of oral drugs, or patients who cannot take oral medications (e.g., coma, vomiting). However, this method carries risks such as infection, phlebitis, and adverse drug reactions, and strict adherence to aseptic operation standards is required."),
    ("CT 检查即计算机断层扫描，利用 X 线束对人体特定部位进行连续断层扫描，结合计算机处理重建出人体断层图像，可清晰显示器官结构、病变位置及性质，常用于头部外伤、肺部疾病（如肺炎、肺癌）、肿瘤排查、骨骼损伤等诊断，不过存在一定电离辐射，需根据病情合理使用。", "CT scan, or Computed Tomography, uses X-ray beams to perform continuous cross-sectional scans of specific parts of the human body. Combined with computer processing, it reconstructs cross-sectional images of the human body, which can clearly show the structure of organs, the location and nature of lesions. It is often used in the diagnosis of head injuries, lung diseases (e.g., pneumonia, lung cancer), tumor screening, and bone injuries. However, it involves a certain amount of ionizing radiation and should be used reasonably based on the patient's condition."),
    ("糖尿病是一组以慢性血糖水平升高为特征的代谢性疾病，主要分为 1 型（胰岛素分泌绝对不足）和 2 型（胰岛素抵抗或分泌相对不足），典型症状为 三多一少（多饮、多食、多尿、体重减轻）。长期高血糖会引发神经病变、肾病、视网膜病变等并发症，治疗需结合饮食控制、运动、血糖监测及药物（如胰岛素、口服降糖药）。", "Diabetes is a group of metabolic diseases characterized by chronic elevated blood glucose levels. It is mainly divided into Type 1 (absolute insulin deficiency) and Type 2 (insulin resistance or relative insulin deficiency). The typical symptoms are the 'three polys and one deficiency' (polydipsia, polyphagia, polyuria, and weight loss). Long-term high blood glucose can cause complications such as neuropathy, nephropathy, and retinopathy. Treatment requires a combination of dietary control, exercise, blood glucose monitoring, and medications (e.g., insulin, oral hypoglycemic drugs)."),
    ("抗生素是一类用于抑制或杀灭细菌的药物，可治疗细菌感染引起的疾病（如肺炎、尿路感染、皮肤感染），但对病毒感染（如感冒、流感）无效。滥用抗生素会导致细菌耐药性，使药物疗效下降甚至失效，临床需根据感染类型、细菌药敏试验结果合理选择与使用。", "Antibiotics are a class of drugs used to inhibit or kill bacteria. They can treat diseases caused by bacterial infections (e.g., pneumonia, urinary tract infections, skin infections) but are ineffective against viral infections (e.g., the common cold, influenza). The abuse of antibiotics can lead to bacterial resistance, reducing the efficacy of drugs or even making them ineffective. In clinical practice, antibiotics should be selected and used reasonably based on the type of infection and the results of bacterial susceptibility tests."),
    ("心电图（ECG）是通过在人体体表放置电极，记录心脏电活动变化的检查方法，能反映心脏节律、心率、心肌供血情况及心脏结构异常（如心肌梗死、心律失常、心肌炎），操作简便、无创，常用于心脏疾病筛查、手术监护、体检等场景，对急性心肌梗死的早期诊断具有重要意义。", "Electrocardiogram (ECG) is an examination method that records changes in cardiac electrical activity by placing electrodes on the surface of the human body. It can reflect the heart rhythm, heart rate, myocardial blood supply, and abnormalities in cardiac structure (e.g., myocardial infarction, arrhythmia, myocarditis). It is easy to operate, non-invasive, and commonly used in scenarios such as cardiac disease screening, surgical monitoring, and physical examinations. It is of great significance for the early diagnosis of acute myocardial infarction."),
    ("脑卒中俗称 中风，因脑部血管突然破裂（出血性脑卒中）或血管阻塞导致血液无法流入大脑（缺血性脑卒中），引起脑组织损伤的急性脑血管疾病。常见症状为突然肢体麻木、言语不清、口角歪斜、意识障碍等，发病后需及时就医（如缺血性脑卒中溶栓治疗），否则易遗留偏瘫、失语等后遗症。", "Stroke, commonly known as 'zhongfeng' in Chinese, is an acute cerebrovascular disease that causes brain tissue damage due to the sudden rupture of blood vessels in the brain (hemorrhagic stroke) or vascular obstruction that prevents blood from flowing into the brain (ischemic stroke). Common symptoms include sudden limb numbness, slurred speech, deviated mouth, and disturbance of consciousness. After the onset, timely medical treatment is required (e.g., thrombolytic therapy for ischemic stroke); otherwise, sequelae such as hemiplegia and aphasia are likely to remain."),
    ("疫苗是用各类病原微生物（如细菌、病毒、立克次体）经过减毒、灭活或基因工程等技术制备的生物制品，接种后可刺激人体免疫系统产生特异性抗体或免疫细胞，当人体再次接触该病原体时，能快速启动免疫反应，预防感染疾病。常见疫苗包括新冠疫苗、乙肝疫苗、脊髓灰质炎疫苗、流感疫苗等，是预防传染病的重要手段。", "Vaccines are biological products prepared from various pathogenic microorganisms (e.g., bacteria, viruses, rickettsia) through techniques such as attenuation, inactivation, or genetic engineering. After vaccination, they can stimulate the human immune system to produce specific antibodies or immune cells. When the human body is exposed to the pathogen again, it can quickly initiate an immune response to prevent infectious diseases. Common vaccines include COVID-19 vaccines, hepatitis B vaccines, polio vaccines, and influenza vaccines, which are important means of preventing infectious diseases.")
    ]
    lines = text.split(" ")
    text = "".join( lines)

    for pair in documents:
        if text == pair[0]:
            return pair[1]
        if text in "".join(pair[1].split(" ")):
            return pair[0]



def create_sample_cedocuments() -> List[str]:
    documents = [
        "Complete Blood Count (CBC) is the most basic clinical blood test. By detecting the quantity and morphological changes of indicators such as red blood cells, white blood cells, platelets, and hemoglobin in the blood, it helps determine the presence of conditions like infections, anemia, and hematological diseases. It is widely used in routine physical examinations, disease diagnosis, and treatment monitoring.",
        "Targeted therapy is a precise anti-cancer treatment method. Based on unique 'targets' of tumor cells such as genetic mutations and protein expression, it uses targeted drugs (e.g., small-molecule targeted drugs, monoclonal antibodies) to act specifically on these targets, blocking the growth signals of tumor cells or destroying them. It causes less damage to normal cells and is commonly used in the treatment of malignant tumors such as lung cancer, breast cancer, and leukemia.",
        "Hypertension is a chronic disease characterized by sustained elevation of systemic arterial blood pressure. The diagnostic criterion is that without the use of antihypertensive drugs, the systolic blood pressure is ≥140 mmHg and/or the diastolic blood pressure is ≥90 mmHg in three non-consecutive measurements on different days. Long-term hypertension can damage target organs such as the heart, brain, kidneys, and eyes, and induce complications like stroke, myocardial infarction, and renal failure. It needs to be controlled for a long time through lifestyle interventions (e.g., low-salt diet, exercise) or medications.",
        "Intravenous infusion is a treatment method that delivers drugs, nutrient solutions, blood products, etc., into the human bloodstream through venous puncture. It allows drugs to take effect quickly and is suitable for patients in emergency situations (e.g., severe infection, shock), patients with poor absorption of oral drugs, or patients who cannot take oral medications (e.g., coma, vomiting). However, this method carries risks such as infection, phlebitis, and adverse drug reactions, and strict adherence to aseptic operation standards is required.",
        "CT scan, or Computed Tomography, uses X-ray beams to perform continuous cross-sectional scans of specific parts of the human body. Combined with computer processing, it reconstructs cross-sectional images of the human body, which can clearly show the structure of organs, the location and nature of lesions. It is often used in the diagnosis of head injuries, lung diseases (e.g., pneumonia, lung cancer), tumor screening, and bone injuries. However, it involves a certain amount of ionizing radiation and should be used reasonably based on the patient's condition.",
        "Diabetes is a group of metabolic diseases characterized by chronic elevated blood glucose levels. It is mainly divided into Type 1 (absolute insulin deficiency) and Type 2 (insulin resistance or relative insulin deficiency). The typical symptoms are the 'three polys and one deficiency' (polydipsia, polyphagia, polyuria, and weight loss). Long-term high blood glucose can cause complications such as neuropathy, nephropathy, and retinopathy. Treatment requires a combination of dietary control, exercise, blood glucose monitoring, and medications (e.g., insulin, oral hypoglycemic drugs).",
        "Antibiotics are a class of drugs used to inhibit or kill bacteria. They can treat diseases caused by bacterial infections (e.g., pneumonia, urinary tract infections, skin infections) but are ineffective against viral infections (e.g., the common cold, influenza). The abuse of antibiotics can lead to bacterial resistance, reducing the efficacy of drugs or even making them ineffective. In clinical practice, antibiotics should be selected and used reasonably based on the type of infection and the results of bacterial susceptibility tests.",
        "Electrocardiogram (ECG) is an examination method that records changes in cardiac electrical activity by placing electrodes on the surface of the human body. It can reflect the heart rhythm, heart rate, myocardial blood supply, and abnormalities in cardiac structure (e.g., myocardial infarction, arrhythmia, myocarditis). It is easy to operate, non-invasive, and commonly used in scenarios such as cardiac disease screening, surgical monitoring, and physical examinations. It is of great significance for the early diagnosis of acute myocardial infarction.",
        "Stroke, commonly known as 'zhongfeng' in Chinese, is an acute cerebrovascular disease that causes brain tissue damage due to the sudden rupture of blood vessels in the brain (hemorrhagic stroke) or vascular obstruction that prevents blood from flowing into the brain (ischemic stroke). Common symptoms include sudden limb numbness, slurred speech, deviated mouth, and disturbance of consciousness. After the onset, timely medical treatment is required (e.g., thrombolytic therapy for ischemic stroke); otherwise, sequelae such as hemiplegia and aphasia are likely to remain.",
        "Vaccines are biological products prepared from various pathogenic microorganisms (e.g., bacteria, viruses, rickettsia) through techniques such as attenuation, inactivation, or genetic engineering. After vaccination, they can stimulate the human immune system to produce specific antibodies or immune cells. When the human body is exposed to the pathogen again, it can quickly initiate an immune response to prevent infectious diseases. Common vaccines include COVID-19 vaccines, hepatitis B vaccines, polio vaccines, and influenza vaccines, which are important means of preventing infectious diseases."
    ]

    return documents





def main( type : bool):
    print("🎯 基础医疗常识RAG系统演示")
    print("=" * 60)

    # 1. 初始化RAG系统
    rag_system = RagDemo()
    cerag_system = RagDemo()

    # 2. 准备样本文档
    print("📋 准备样本文档...")
    documents = create_sample_documents()
    cddocuments = create_sample_cedocuments()

    # 3. 添加文档到知识库
    rag_system.add_documents(documents)

    cerag_system.add_documents(cddocuments)

    print("\n" + "=" * 60)

    # 4. 交互式查询（可选）
    print("\n" + "=" * 60)
    print("💬 进入交互模式（输入 'quit' 退出）:")

    if type:
        while True:
            user_query = input("\n请输入您的问题: ").strip()

            if user_query.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break

            if user_query:
                try:
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', user_query))
                    english_chars = len(re.findall(r'[a-zA-Z]', user_query))
                    if chinese_chars < english_chars:
                        answer = cerag_system.rag_pipeline(user_query)
                        answerTranslation = conversion(answer)
                        if not answer:
                            print("\n💡抱歉，没有找到相关的信息来回答您的问题。")
                            continue
                    else :
                        answer = rag_system.rag_pipeline(user_query)
                        answerTranslation = conversion(answer)
                        if not answer:
                            print("\n💡抱歉，没有找到相关的信息来回答您的问题。")
                            continue
                    print(f"\n💡 \n 回答:{answer} \n 译文: {answerTranslation}")
                except Exception as e:
                    print(f"❌ 处理查询时出错: {e}")
    else:
        batch_texts = [
            "Stroke",
            "血常规是临床最基础的血液检验项目",
            "抗生素是什么"
        ]

        for query in batch_texts:
            try:
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
                english_chars = len(re.findall(r'[a-zA-Z]', query))
                if chinese_chars < english_chars:
                    answer = cerag_system.rag_pipeline(query)
                    answerTranslation = conversion(answer)
                    if not answer:
                        print("\n💡抱歉，没有找到相关的信息来回答您的问题。")
                        continue
                else:
                    answer = rag_system.rag_pipeline(query)
                    answerTranslation = conversion(answer)
                    if not answer:
                        print("\n💡抱歉，没有找到相关的信息来回答您的问题。")
                        continue
                print(f"\n💡 回答:{answer} \n💡 译文: {answerTranslation}")
            except Exception as e:
                print(f"❌ 处理查询时出错: {e}")


if __name__ == "__main__":
    main(False)
