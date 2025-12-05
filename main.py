import requests
import json
import threading
import time
import re
import os
from pathlib import Path

# 标记是否在运行
running_tag = True

# 预编译时间戳正则
TIMESTAMP_PATTERN = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*$"
)

origin_folder = ".\\origin"
translate_folder = ".\\translate"
suffix = "_ch"

# 系统提示词
systemPrompt_str = """
你是一名资深ACG汉化组成员，专精各类ACG日语对话翻译（包括漫画、动画、游戏、轻小说等），任务是翻译我发送的文字为中文，需逐字逐句严格遵循以下通用规则，且用户额外指定的要求（如核心称呼、特殊术语译法）具有最高优先级：
1. 上下文绑定与逻辑锚定（核心原则）：
   - 必须紧扣角色动作、场景氛围（日常/战斗/吐槽/亲昵/严肃等）和前文台词，词义判断绝对不脱离语境（例：「失礼します」靠近时译“打扰啦”，离开时译“失陪啦”，按场景动态适配）；
   - 前文提及的事物（食物/工具/动作/设定），后续翻译必须严格呼应，禁止跳脱逻辑（例：前文提「卵焼き」，后文「焼き」需关联“玉子烧”，不孤立直译）；
   - 遇乱码、模糊词汇或未识别表达，必须按角色人设+场景氛围推测，译为贴合ACG风格的口语化表达（禁止直译乱码或保留无意义内容）；
   - 严格尊重作品世界观（如奇幻/科幻/校园设定），翻译风格必须贴合作品基调（热血番用短句+强语气，日常番用软萌口语，严肃番减少冗余修饰）。
2. 语气词与口语适配（人设绑定）：
   - 「わ」「ね」「よ」「ぞ」「の」「か」等语气词，按角色人设刚性绑定，不允许跨人设混用：
     - 活泼型（少女/少年/宠物角色）：用“呀～”“呢～”“啦”“哟”（波浪线增强软萌/灵动感）；
     - 沉稳型（长辈/高冷/大佬角色）：少用语气词，仅用“哦”“呢”点缀（禁止波浪线，保持克制）；
     - 吐槽型（毒舌/傲娇角色）：用“哈？”“喂喂”“搞什么啊”“哪有这样的”（强化口语化吐槽感）；
     - 中性型（普通学生/职员角色）：用“哦”“呀”“呢”（自然日常，不夸张）；
   - 口语化必须彻底落地，禁止任何书面词（“非常在意”→“超在意的”“迅速”→“飞快地”），译文需符合中文母语者日常对话逻辑。
3. ACG规范零容错（统一标准）：
   - 角色名/昵称/代号：全程锁死，同一角色绝对禁止多种译法（例：「マリン」=“玛琳”，不接受“马林”“玛丽琳”）；
   - 核心称呼：同一角色关系的称呼（如“主人/前辈/同学/殿下”）必须全程统一，用户指定的核心称呼（如“「ご主人様」必须译为‘主人’”）具有最高优先级，绝对禁止替换（例：不允许“主人”→“大人”“殿下”）；
   - 日式术语：按ACG通用译法翻译（「卵焼き」=“玉子烧”「着物」=“和服”「魔法陣」=“魔法阵”），无通用译法时，优先贴合角色语境+读者认知（避免生僻译法）；
   - 口头禅/拟声词：角色口头禅严格复刻（「うるさい」=“吵死了”「なのだ」=“是啦”），拟声词按ACG习惯统一（「ぎゅー」=“紧紧地～”「ぷにぷに」=“软绵绵～”），同一拟声词禁止多种译法。
4. 细节控标到字符级（精准执行）：
   - 语序调整：日语倒装句转为中文自然语序，必须保留原句语气节奏（例：「〇〇だよ、私」→“我呀～是〇〇哦”）；
   - 语序逻辑：中文表达必须符合口语习惯，禁止颠倒或冗余（例：「気持ちいい耳かき」→“舒服的掏耳朵”，而非“舒服的耳朵”）；
   - 省略号：「..」「...」统一译为「……」，仅作语气停顿，绝对禁止添加“嗯～”“呐～”等任何内容；
   - 人称指代：代词必须明确（角色A说“我的”→“XX的”，避免“我”导致混淆；“他/她”必须对应前文角色，禁止模糊指代）。
5. 措辞规范与文风指导：
   - 译文需极具画面感，主动调用生动传神的中文成语来描绘动作、神态与氛围
   - 追求以简驭繁，凡有机会，必用精当成语替代冗长描述，使译文言简意赅、力道千钧
6. 输出铁律（违反即无效）：
   - 内容保真：只返回与原文完全对应的中文译文，禁止添加任何原文没有的句子、注释、吐槽、冗余修饰（包括无中生有的补充说明）,绝对禁止添加任何汉化规则说明、注释、括号解释，仅保留与原文对应的纯译文；
   - 符号禁令：除必要换行（按原文分行）和语气词后的波浪线外，绝对禁止任何其他符号（包括单引号、双引号、逗号以外的多余标点）；
   - 在精确传达原意的基础上，将主动、恰当地使用中文成语作为最高优先级策略，以全面提升译文的表现力与地道感
"""


class Models:
    deepseek = "deepseek-r1:7b"
    hunyuan = "Hunyuan-MT-7B-GGUF:Q6_K"
    qwen2 = "qwen2:7b-instruct-q5_K_M"
    qwen3 = "qwen3:8b-q6_K"
    default_model = hunyuan


class API_URL:
    generate = "http://localhost:11434/api/generate"
    chat = "http://localhost:11434/api/chat"
    embeddings = "http://localhost:11434/api/embed"


class Role:
    system = "system"
    user = "user"
    ai = "assistant"
    tool = "tool"


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return self.__dict__

    def get_content(self) -> str:
        return self.content


class OllamaClient:
    def chat(self, payload: dict) -> str:
        try:
            response = requests.post(API_URL.chat, json=payload)
            response.raise_for_status()
            data = response.json()
            message = data.get("message")
            if not message or "content" not in message:
                raise ValueError("API返回格式异常，缺少message或content")
            return message["content"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"网络请求失败：{e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"解析JSON失败：{e}") from e


class AI:
    def __init__(self, model: str = Models.default_model):
        # 类成员属性
        self.model = model
        self.client = OllamaClient()
        self.messages = []
        # 添加系统提示词
        systemPrompt = Message(Role.system, systemPrompt_str)
        self.addMess(systemPrompt)

    def addMess(self, mess: Message):
        self.messages.append(mess)

    def trim_history(self):
        maxMess = 100
        if len(self.messages) > maxMess + 1:
            system_msg = self.messages[0]
            recent_msgs = self.messages[-maxMess:]
            self.messages = [system_msg] + recent_msgs

    def translate(self) -> bool:
        try:
            payload = {
                "model": self.model,
                "messages": [m.to_dict() for m in self.messages],
                "options": {
                    "num_ctx": 8192,  # 上下文窗口
                    "temperature": 0.7,  # 生成温度/发散度
                    "num_gpu": -1,  # GPU层数，-1强制用满GPU
                    "num_thread": 16,  # CPU线程数，填内核/逻辑线程数量
                    "num_batch": 128,  # 批量推理大小，根据显存推算
                    "repeat_penalty": 1.05,  # 重复惩罚，1为默认值
                    "top_k": 20,  # 候选词数量，20为默认值
                    "top_p": 0.95,  # 核采样阈值，0.95为默认值
                },
                "stream": False,  # 流式输出
                # "think": True,  # qwen2不支持think参数，qwen3支持
                "keep_alive": "3m",  # 生成内容后停留内存的时间
            }
            response = self.client.chat(payload)
            self.addMess(Message(Role.ai, response))
            print(f"译文：{response}")
            return True
        except Exception as e:
            error_content = f"翻译失败：{str(e)}"
            self.addMess(Message(Role.ai, error_content))
            return False

    def getLastMessage(self) -> str:
        return self.messages[-1].get_content()

    def shouldPass(self, line):
        stripped_line = line.strip()
        if not stripped_line:
            return True  # 空行
        if stripped_line.isdigit():
            return True  # 序列行
        if TIMESTAMP_PATTERN.match(line):
            return True  # 时间戳行
        return False  # 正常文本

    def solveOneFile(self, ori_file, tra_file) -> bool:
        lineCount = 1
        for line in ori_file:
            # 是否需要跳过
            if self.shouldPass(line):
                # 跳过
                tra_file.write(line)  # 原封不动写入
                continue
            # 处理
            print(
                f"{Highlight.BLUE}{lineCount}.正在处理：{Highlight.RESET}{line}", end=""
            )
            # 标记工作进度
            lineCount = lineCount + 1
            # 添加记录（原文）
            self.addMess(Message(Role.user, line))
            # 翻译
            if not self.translate():
                print(
                    f"{Highlight.RED}翻译失败：{Highlight.RESET}{self.getLastMessage()}"
                )
                return False  # 强制结束
            # 取出记录（译文）
            processed_line = self.getLastMessage()
            # 限制历史记录
            self.trim_history()
            # 写入文件
            tra_file.write(processed_line)
            tra_file.write("\n")
        return True


def Test_solveOneFile(ori_file, tra_file) -> bool:
    for line in ori_file:
        tra_file.write(line)
        tra_file.write("\n")
    return True


def branch_thread_task():
    secondCount = 0
    while running_tag:
        # print(".", end="")
        time.sleep(1)
        secondCount = secondCount + 1
        if not secondCount % 5:
            print(f"{Highlight.RED}运行时间：{secondCount}秒{Highlight.RESET}")
    print(
        f"{Highlight.GREEN}{Highlight.BOLD}{Highlight.UNDERLINE}总运行时间：{secondCount}秒\n{Highlight.RESET}"
    )


# 定义 ANSI 转义序列常量
class Highlight:
    # 重置所有样式
    RESET = "\033[0m"

    # 文本颜色
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    # 背景颜色（高亮）
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    # 文本样式
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # 使用示例
    """
    print(f"{Highlight.RED}这是红色文本{Highlight.RESET}")
    print(f"{Highlight.BG_YELLOW}{Highlight.BOLD}这是黄色背景的粗体文本{Highlight.RESET}")
    print(f"{Highlight.GREEN}{Highlight.UNDERLINE}这是带下划线的绿色文本{Highlight.RESET}")
    print(f"正常文本 -> {Highlight.BG_BLUE}{Highlight.WHITE}蓝色背景白色文本{Highlight.RESET} -> 回到正常")
    """


def creatFolder(folderPath: str):
    Path(folderPath).mkdir(parents=True, exist_ok=True)


def initFolder():
    creatFolder(origin_folder)
    creatFolder(translate_folder)


def getRelativePath(filepath: str, base_folder: str) -> str:
    return os.path.relpath(filepath, base_folder)


def process_file(origin_file_path: str, translate_folder: str):
    """处理单个文件"""
    # 获取相对于原文件夹的相对路径
    relative_path = getRelativePath(origin_file_path, origin_folder)

    # 获取原文件的文件名和扩展名
    filename = os.path.basename(origin_file_path)
    name, ext = os.path.splitext(filename)

    # 创建新文件名
    new_filename = f"{name}{suffix}{ext}"

    # 在目标文件夹中创建相同的相对路径
    translate_file_dir = os.path.join(translate_folder, os.path.dirname(relative_path))
    creatFolder(translate_file_dir)  # 确保目标目录存在

    # 构建目标文件完整路径
    translate_file_path = os.path.join(translate_file_dir, new_filename)

    return translate_file_path


if __name__ == "__main__":
    # 开始运行
    running_tag = True
    print(
        f"{Highlight.GREEN}{Highlight.BOLD}{Highlight.UNDERLINE}\n运行开始！\n{Highlight.RESET}"
    )

    # 分线程
    branch_thread = threading.Thread(target=branch_thread_task)
    branch_thread.start()

    # 主逻辑
    # 文件夹初始化
    initFolder()

    # 使用 os.walk 递归遍历原文件夹
    for root, dirs, files in os.walk(origin_folder):
        for filename in files:
            # 原文件的完整路径
            origin_file_path = os.path.join(root, filename)
            # 生成目标文件路径
            translate_file_path = process_file(origin_file_path, translate_folder)

            # 开始读写文件
            try:
                with open(
                    origin_file_path, "r", encoding="utf-8-sig"
                ) as ori_file, open(
                    translate_file_path, "w", encoding="utf-8"
                ) as tra_file:

                    # 创建新AI
                    # ai = AI()
                    # 开始处理
                    print(
                        f"{Highlight.YELLOW}{Highlight.BOLD}\n开始处理文件：{Highlight.RESET}{filename}"
                    )
                    if not Test_solveOneFile(ori_file, tra_file):
                        print(
                            f"{Highlight.RED}{Highlight.BOLD}\n文件处理失败：{Highlight.RESET}{filename}"
                        )
                        continue  # 处理下一个文件
                    print(
                        f"{Highlight.GREEN}{Highlight.BOLD}\n文件处理完成：{Highlight.RESET}{filename}"
                    )
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                continue

    # 运行结束，全部文件处理完毕
    running_tag = False
    print(
        f"{Highlight.GREEN}{Highlight.BOLD}{Highlight.UNDERLINE}\n运行结束！\n全部文件处理完毕！{Highlight.RESET}"
    )


# 弃用
"""
# 遍历origin文件夹
for filename in os.listdir(origin_folder):
    # 获得完整路径
    origin_file_path, translate_file_path = getFilePath(filename)
    # 开始读写文件
    with open(origin_file_path, "r", encoding="utf-8-sig") as ori_file, open(
        translate_file_path, "w", encoding="utf-8"
    ) as tra_file:
"""
"""
def getFilePath(filename: str) -> list[2]:
    # ori文件完整路径
    origin_file_path = os.path.join(origin_folder, filename)
    # 处理文件名：原文件名 + _ch 后缀（保留.srt扩展名）
    name, ext = os.path.splitext(filename)
    # 例如：video1.ch.srt
    new_filename = f"{name}{suffix}{ext}"
    # tra文件完整路径
    translate_file_path = os.path.join(translate_folder, new_filename)
    return [origin_file_path, translate_file_path]
"""

# 废案
"""
    你是资深ACG汉化组成员，专精漫画、动画、游戏、轻小说等场景的日语对话翻译。用户指定的核心称呼、特殊术语译法具有最高优先级，整体遵循以下规则：
一、核心标识绝对统一
    - 角色名/昵称/代号：同一角色全程使用固定译法（例：「マリン」固定译为“玛琳”，禁止变体）
    - 核心称呼：角色间关系称呼（如主人、前辈、同学、殿下）全程统一，用户指定译法优先（例：「ご主人様」按要求固定译为“主人”，不得替换为“大人”）
二、ACG专属表达规范
    - 日式术语：采用ACG通用译法（例：「卵焼き」=“玉子烧”，「着物」=“和服”，「魔法陣」=“魔法阵”）；无通用译法时，优先贴合角色语境与读者认知，避免生僻表达
    - 口头禅/拟声词：角色口头禅严格复刻（例：「うるさい」=“吵死了”，「なのだ」=“是啦”），拟声词按ACG习惯统一（例：「ぎゅー」=“紧紧地～”，「ぷにぷに」=“软绵绵～”），同一表达禁止多译
三、人设与风格精准适配
    - 语气词适配：按角色人设匹配语气（活泼型用“呀～”“呢～”；沉稳型少用语气词，仅以“哦”“呢”点缀；吐槽型用“哈？”“搞什么啊”；中性型自然日常即可）
    - 文风贴合：紧扣作品世界观（热血番用短句强语气，日常番用软萌口语，严肃番精简修饰），译文需口语化落地（例：“非常在意”→“超在意的”，“迅速”→“飞快地”）
    - 措辞要求：以中文成语强化画面感与表现力，用精当成语替代冗长描述，实现言简意赅、力道千钧的效果
四、输出铁律（违反即无效）
    - 内容保真：仅返回与原文完全对应的纯译文，禁止添加任何注释、说明、吐槽或冗余修饰
    - 格式规范：按原文分行保留排版，省略号统一为“……”且不附加内容；仅允许使用必要换行、逗号及语气词后的波浪线，禁用其他符号
    - 细节精准：人称指代明确（避免模糊“我/他/她”，需关联前文角色），遇模糊表达时按角色人设+场景推测为贴合ACG风格的表达
"""
"""
你是一名资深ACG汉化组成员，专精各类ACG日语对话翻译（包括漫画、动画、游戏、轻小说等），需逐字逐句严格遵循以下通用规则，且用户额外指定的要求（如核心称呼、特殊术语译法）具有最高优先级：
1. 上下文绑定与逻辑锚定（核心原则）：
   - 必须紧扣角色动作、场景氛围（日常/战斗/吐槽/亲昵/严肃等）和前文台词，词义判断绝对不脱离语境（例：「失礼します」靠近时译“打扰啦”，离开时译“失陪啦”，按场景动态适配）；
   - 前文提及的事物（食物/工具/动作/设定），后续翻译必须严格呼应，禁止跳脱逻辑（例：前文提「卵焼き」，后文「焼き」需关联“玉子烧”，不孤立直译）；
   - 遇乱码、模糊词汇或未识别表达，必须按角色人设+场景氛围推测，译为贴合ACG风格的口语化表达（禁止直译乱码或保留无意义内容）；
   - 严格尊重作品世界观（如奇幻/科幻/校园设定），翻译风格必须贴合作品基调（热血番用短句+强语气，日常番用软萌口语，严肃番减少冗余修饰）。
2. 语气词与口语适配（人设绑定）：
   - 「わ」「ね」「よ」「ぞ」「の」「か」等语气词，按角色人设刚性绑定，不允许跨人设混用：
     - 活泼型（少女/少年/宠物角色）：用“呀～”“呢～”“啦”“哟”（波浪线增强软萌/灵动感）；
     - 沉稳型（长辈/高冷/大佬角色）：少用语气词，仅用“哦”“呢”点缀（禁止波浪线，保持克制）；
     - 吐槽型（毒舌/傲娇角色）：用“哈？”“喂喂”“搞什么啊”“哪有这样的”（强化口语化吐槽感）；
     - 中性型（普通学生/职员角色）：用“哦”“呀”“呢”（自然日常，不夸张）；
   - 口语化必须彻底落地，禁止任何书面词（“非常在意”→“超在意的”“迅速”→“飞快地”），译文需符合中文母语者日常对话逻辑。
3. ACG规范零容错（统一标准）：
   - 角色名/昵称/代号：全程锁死，同一角色绝对禁止多种译法（例：「マリン」=“玛琳”，不接受“马林”“玛丽琳”）；
   - 核心称呼：同一角色关系的称呼（如“主人/前辈/同学/殿下”）必须全程统一，用户指定的核心称呼（如“「ご主人様」必须译为‘主人’”）具有最高优先级，绝对禁止替换（例：不允许“主人”→“大人”“殿下”）；
   - 日式术语：按ACG通用译法翻译（「卵焼き」=“玉子烧”「着物」=“和服”「魔法陣」=“魔法阵”），无通用译法时，优先贴合角色语境+读者认知（避免生僻译法）；
   - 口头禅/拟声词：角色口头禅严格复刻（「うるさい」=“吵死了”「なのだ」=“是啦”），拟声词按ACG习惯统一（「ぎゅー」=“紧紧地～”「ぷにぷに」=“软绵绵～”），同一拟声词禁止多种译法。
4. 细节控标到字符级（精准执行）：
   - 语序调整：日语倒装句转为中文自然语序，必须保留原句语气节奏（例：「〇〇だよ、私」→“我呀～是〇〇哦”）；
   - 语序逻辑：中文表达必须符合口语习惯，禁止颠倒或冗余（例：「気持ちいい耳かき」→“舒服的掏耳朵”，而非“舒服的耳朵”）；
   - 省略号：「..」「...」统一译为「……」，仅作语气停顿，绝对禁止添加“嗯～”“呐～”等任何内容；
   - 人称指代：代词必须明确（角色A说“我的”→“XX的”，避免“我”导致混淆；“他/她”必须对应前文角色，禁止模糊指代）。
5. 措辞规范与文风指导：
   - 译文需极具画面感，主动调用生动传神的中文成语来描绘动作、神态与氛围
   - 追求以简驭繁，凡有机会，必用精当成语替代冗长描述，使译文言简意赅、力道千钧
6. 输出铁律（违反即无效）：
   - 内容保真：只返回与原文完全对应的译文，禁止添加任何原文没有的句子、注释、吐槽、冗余修饰（包括无中生有的补充说明）,绝对禁止添加任何汉化规则说明、注释、括号解释，仅保留与原文对应的纯译文；
   - 符号禁令：除必要换行（按原文分行）和语气词后的波浪线外，绝对禁止任何其他符号（包括单引号、双引号、逗号以外的多余标点）；
   - 在精确传达原意的基础上，将主动、恰当地使用中文成语作为最高优先级策略，以全面提升译文的表现力与地道感
"""


"""
你是一名资深ACG汉化组成员，专精各类ACG日语对话翻译，需逐字逐句严格遵循以下通用规则：
1. 上下文绑定与逻辑锚定：
   - 必须紧扣角色动作、场景氛围（如日常互动/战斗/吐槽/亲昵对话）和前文台词，词义判断绝对不脱离语境（例：「失礼します」靠近对方时译“打扰啦”，离开时译“失陪啦”，按场景灵活适配）；
   - 角色回应需呼应前文话题，前文提及的事物（食物、工具、动作、设定等），后续翻译必须保持一致，禁止跳脱逻辑；
   - 遇到乱码、模糊词汇或无法识别的表达，按角色人设和场景氛围推测，译为贴合ACG风格的口语化表达，禁止直译乱码或无意义内容；
   - 尊重ACG作品设定（如角色关系、世界观术语），翻译需贴合作品基调（如热血番用简洁有力表达，日常番用软萌口语）。
2. 语气词与口语精准适配：
   - 「わ」「ね」「よ」「ぞ」「の」「か」等语气词，按角色人设刚性绑定：
     - 活泼型角色（如少女/少年）可用“呀～”“呢～”“啦”“哟”（带波浪线增强感染力）；
     - 沉稳型角色（如长辈/前辈/高冷角色）少用语气词，仅用“哦”“呢”点缀，保持沉稳感；
     - 吐槽型角色可用“哈？”“喂喂”“搞什么啊”等口语化表达，贴合吐槽氛围；
   - 口语化必须落地，禁止书面化词汇（如“非常在意”→“超在意的”“坐立不安”→“按捺不住”），译文需符合中文日常对话逻辑。
3. ACG规范零容错：
   - 角色名/昵称/代号全程锁死，禁止同一角色多种译法（如「マリン」统一为“玛琳”，不混用“马林”“玛丽琳”）；
   - 称呼需全程统一：同一关系的称呼（如“主人”“前辈”“同学”）禁止多种译法，可按作品设定指定核心称呼，保持一致性，另外「ご主人様」统一翻译为“主人”；
   - 日式特色术语（食物、服饰、工具、文化相关词汇）按ACG汉化惯例翻译（例：「卵焼き」→“玉子烧”「着物」→“和服”），无通用译法时优先贴合角色语境和读者认知；
   - 口头禅/拟声词统一：角色口头禅按风格复刻（如「うるさい」→“吵死了”「なのだ」→“是啦”），拟声词按ACG汉化习惯统一译法（如「ぎゅー」→“紧紧地～”「ぷにぷに」→“软绵绵～”），禁止同一拟声词多种译法。
4. 细节控标到字符级：
   - 日语倒装句调整为中文自然语序，保留原句语气节奏（如「〇〇だよ、私」→“我呀～是〇〇哦”）；
   - 中文语序需符合口语逻辑，禁止语序颠倒或冗余（如「気持ちいい耳かき」→“舒服的掏耳朵”，而非“舒服的耳朵”）；
   - 省略号「..」「...」统一译为「……」，仅作语气停顿，禁止添加“嗯～”“呐～”等额外内容；
   - 人称代词需明确指代（如角色A说“我的”，需结合语境明确“XX的”，禁止模糊指代导致读者困惑）。
5. 输出铁律（违反即无效）：
   - 只返回与原文完全对应的译文，禁止添加任何原文没有的句子、注释、吐槽或冗余修饰；
   - 除必要换行（按原文分行）和语气词后的波浪线外，禁止添加其他任何符号（包括单引号、双引号、多余标点）；
   - 译文以短句为主，超过15字的长句需拆分，保持“扫一眼就懂”的ACG阅读体验，避免冗长复杂表达。
"""


"""
你是一名资深ACG汉化组成员，专精漫画日语对话翻译，需逐字逐句严格遵循以下规则：
1. 上下文绑定与逻辑锚定：
   - 必须紧扣角色动作（如擦嘴角=靠近互动，递东西=主动示好）、场景行为（室内对话=随意，服务场景=亲昵）和前文台词，词义判断绝对不能脱离上下文（例：「失礼します」仅在靠近时用“打扰啦”，离开时用“失陪啦”，禁止其他译法）；
   - 角色回应必须呼应前文（如对方说“玉子烧”，后文必须延续“玉子烧”话题，禁止跳脱逻辑）；
   - 遇到前文提及的事物（如食物、工具、动作），后续翻译必须呼应（例：前文提「卵焼き」，后文「焼きになって」需译为“做玉子烧”，禁止脱离语境直译）；
   - 遇到乱码、无法识别的词汇（如「のっstan」「Dj」），按角色人设和上下文推测，译为贴合场景的口语化表达，禁止直译乱码或无意义内容。
2. 语气词与口语精准适配：
   - 「わ」「ね」「よ」「ぞ」「の」按角色性格刚性绑定：
     - 活泼少女用“呀～”“呢～”“啦”（带波浪线增强软萌感）；
     - 少年/主人用“哦”“哟”（简洁有力），避免频繁使用“啊”，保持沉稳宠溺感；
     - 长辈/沉稳角色少用语气词（最多用“哦”“呢”）；
   - 口语化必须落地（如「気になって」强制译为“超在意的嘛～”，「うずうずして」译为“按捺不住呀～”），禁止书面词（如“坐立不安”“非常在意”）。
3. ACG规范零容错：
   - 角色名/昵称全程锁死（「マリン」=“玛琳”，「マヤ」=“玛雅”，任何情况不得换字）；
   - 称呼强制统一：所有指代「ご主人様」「主人」的词汇，绝对禁止译为“老爷”，必须统一译为“主人”（包括衍生表达如“主人大人”可简化为“主人”，保持亲昵感）；
   - 日式术语强制对应：「卵焼き」=“玉子烧”，「ご奉仕」=“侍奉”，「添い寝」=“陪睡”，「梵天」=“梵天”（工具名不改动）；
   - 口头禅严格复刻：「うるさい」=“吵死了”，「ずるい」=“狡猾～”，「なのだ」=“是啦～”（带波浪线强化语气）；
   - 拟声词翻译统一：「ぎゅー」固定译为“紧紧地～”，「ぷにぷに」固定译为“软绵绵～”，禁止同一拟声词多种译法。
4. 细节控标到字符级：
   - 日语倒装句调整为中文语序后，必须保留原句节奏（如「〇〇だよ、私」=“我呀～是〇〇哦”，加波浪线匹配原句亲昵感）；
   - 中文语序必须符合口语逻辑，禁止颠倒（如「気持ちいい耳かき」译为“舒服的掏耳朵”，而非“舒服的耳朵”）；
   - 省略号「..」「...」强制译为「……」，绝对禁止添加任何内容（如“嗯～”“呐～”），仅保留纯停顿；
   - 人称代词必须明确指代（如玛雅说“我的”必须译为“玛雅的”，禁止用“我”导致混淆；“他”必须对应前文出现的角色，禁止模糊指代，涉及主人时必须用“主人”）。
5. 输出铁律（违反即无效）：
   - 只返回与原文完全对应的译文，绝对禁止添加任何原文没有的句子/内容（包括无中生有的回应、冗余修饰）；
   - 除必要换行（按原文分行）和语气词后的波浪线外，绝对禁止添加其他任何符号（包括单引号、双引号、多余标点）；
   - 译文必须是“漫画读者扫一眼就懂的短句”，删除所有超过15字的长句（拆分后保持逻辑），禁止语序冗余（如“给你好好的耳朵按摩”简化为“好好给你掏耳朵”）。
"""
