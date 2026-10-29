"""Names as data (live voice, LV2): how a name the cloud model proposes becomes (or does not become)
a name the duck keeps.

The model writes a name as it heard it, and the transcript it hears from is not reliable for names
(LV1: 蛋黃 was heard as 郑华 in all 8 ``call_name`` recordings; 妈妈 and 媽媽 both came back for
the same introduction). So before a name is stored:

* it is **normalised** (:func:`normalise_name`): one form for the same name — 简体 to 繁體 for the
  characters a family name or a given name is made of, no spaces, punctuation or quotation
  marks, no "這是" in front of it;
* it is compared with the names the duck already has — **its own name** and the **family-name
  list** of its learner (:func:`decide_name`) — by characters and by sound (:func:`similarity`;
  a small pinyin table for the characters of names, a heuristic: a character that is not in it
  compares by character only), so a name that is only a mishearing of the duck's own name, or
  another way of writing someone it knows, is not made into a new person;
* it is bound only to **a person the duck itself sees right now**, and only when a person has just
  spoken; when it is not sure the name is kept as *pending* on that person (never as a new person,
  never invented), and two separate pieces of evidence are needed before a pending *rename* is
  taken.

Nothing here reads the transcript, a world id or the model's other words: a name in, a decision
out. The characters of a name are data and are never translated.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

MAX_NAME_CHARS = 6  # 歐陽小明 is 4; a "name" longer than this is a sentence
MAX_LATIN_NAME = 14
MAX_VOCABULARY = (
    12  # the phrases the transcriber is biased toward: the duck's name and the family's
)
SIM_OWN_NAME = 0.6  # this alike the duck's own name: a mishearing of it, not a person
SIM_KNOWN_NAME = 0.75  # this alike a name it knows: the same person, written another way
PROMOTE_AT = 2  # evidence a pending rename needs
RECENT_SPEECH_S = 6.0  # a person must have spoken this recently for a name to be taken at once

# simplified → traditional, for what names and family words are made of (not a full converter:
# a character that is the same in both, or not here, is kept as it is)
_PAIRS = (
    # family words and surnames
    "妈媽 爷爺 孙孫 儿兒 宝寶 贝貝 亲親 这這 饭飯 鸭鴨 华華 黄黃 郑鄭 张張 刘劉 陈陳 杨楊 吴吳 赵趙 "
    "马馬 钱錢 汤湯 萧蕭 邓鄧 许許 吕呂 苏蘇 卢盧 蒋蔣 贾賈 叶葉 阎閻 钟鍾 谭譚 邹鄒 陆陸 龙龍 万萬 "
    "乔喬 贺賀 赖賴 龚龔 顾顧 "
    # given names
    "丽麗 伟偉 强強 静靜 艳艷 军軍 红紅 云雲 国國 爱愛 欢歡 乐樂 凤鳳 兰蘭 岚嵐 颖穎 莹瑩 诗詩 涛濤 "
    "鹏鵬 飞飛 晓曉 骏駿 继繼 "
    # what a sentence around a name is made of
    "们們 个個 么麼 说說 谁誰 请請 问問 为為 会會 点點 来來 过過 还還 时時 听聽 见見 觉覺 让讓 认認 "
    "识識 难難 题題 长長 门門 间間 师師 学學"
)
S2T: dict[str, str] = {pair[0]: pair[1] for pair in _PAIRS.split()}

# ── a transcript, shown in the family's own script ───────────────────────────────────
#
# The transcriber is told the family speaks zh-TW (``Setup.transcription_languages``), which is
# what keeps most of a transcript in traditional characters; this is what is left for a sentence
# that still comes back simplified. It changes only what the page SHOWS: the log keeps the raw text.
# The table is a character-by-character one (ICU's Hans-Hant, the characters of GB2312's first
# level: what people say), without the characters where one simplified form stands for several
# traditional ones or is itself a traditional character (干 复 征 占 于 冲 舍 范 采 …): those are left
# as they came, never guessed. A handful of words that need more than a character (头发 is 頭髮, not
# 頭發; 这里 is 這裡) have their own rule below.
_TEXT_PAIRS = (
    "万萬 与與 专專 业業 丛叢 东東 丝絲 丢丟 两兩 严嚴 丧喪 个個 丰豐 临臨 为為 丽麗 举舉 么麼 义義 乌烏 乐樂 乔喬 习習 乡鄉 书書 买買 乱亂 争爭 亏虧 云雲 亚亞 "
    "产產 亩畝 亲親 亿億 仅僅 仆僕 从從 仑侖 仓倉 仪儀 们們 价價 众眾 优優 会會 伞傘 伟偉 传傳 伤傷 伦倫 伪偽 体體 佣傭 侠俠 侣侶 侥僥 侦偵 侧側 侨僑 侩儈 俩倆 "
    "俭儉 债債 倾傾 偿償 储儲 儿兒 兑兌 党黨 兰蘭 关關 兴興 兹茲 养養 兽獸 内內 冈岡 册冊 写寫 军軍 农農 冯馮 决決 况況 冻凍 净淨 凄淒 凉涼 减減 凑湊 凛凜 几幾 "
    "凤鳳 凭憑 凯凱 击擊 凿鑿 刘劉 则則 刚剛 创創 删刪 别別 刹剎 刽劊 剂劑 剐剮 剑劍 剥剝 剧劇 劝勸 办辦 务務 动動 励勵 劲勁 劳勞 势勢 勋勳 匀勻 区區 医醫 华華 "
    "协協 单單 卖賣 卢盧 卤鹵 卧臥 卫衛 却卻 厂廠 厅廳 历歷 厉厲 压壓 厌厭 厕廁 厢廂 厦廈 厨廚 厩廄 县縣 叁叄 参參 双雙 发發 变變 叙敘 叠疊 叶葉 号號 叹嘆 后後 "
    "吓嚇 吕呂 吗嗎 吨噸 听聽 启啟 吴吳 呐吶 呕嘔 员員 呛嗆 呜嗚 咏詠 咙嚨 响響 哑啞 哗嘩 哟喲 唤喚 啮嚙 啸嘯 喷噴 嘘噓 嘱囑 嚣囂 团團 园園 囱囪 围圍 国國 图圖 "
    "圆圓 圣聖 场場 坏壞 块塊 坚堅 坛壇 坝壩 坞塢 坟墳 坠墜 垄壟 垒壘 垦墾 垫墊 堑塹 堕墮 墙牆 壮壯 声聲 壳殼 壶壺 处處 备備 够夠 头頭 夸誇 夹夾 夺奪 奋奮 奖獎 "
    "奥奧 妆妝 妇婦 妈媽 娄婁 娇嬌 娱娛 婴嬰 婶嬸 孙孫 学學 孪孿 宁寧 宝寶 实實 宠寵 审審 宪憲 宫宮 宽寬 宾賓 寝寢 对對 寻尋 导導 寿壽 将將 尔爾 尘塵 尝嘗 尧堯 "
    "尸屍 尽盡 层層 屉屜 届屆 属屬 屡屢 屿嶼 岁歲 岂豈 岗崗 岛島 岭嶺 岿巋 峡峽 峦巒 崭嶄 巩鞏 币幣 帅帥 师師 帐帳 帘簾 帜幟 带帶 帧幀 帮幫 幂冪 广廣 庄莊 庆慶 "
    "庐廬 库庫 应應 庙廟 庞龐 废廢 开開 异異 弃棄 张張 弥彌 弯彎 弹彈 强強 归歸 当當 录錄 彦彥 彻徹 径徑 忆憶 忧憂 怀懷 态態 怂慫 怜憐 总總 恋戀 恒恆 恳懇 恶惡 "
    "恼惱 悦悅 悬懸 悯憫 惊驚 惧懼 惨慘 惩懲 惫憊 惭慚 惮憚 惯慣 愤憤 愿願 慑懾 懒懶 戏戲 战戰 户戶 扑撲 执執 扩擴 扫掃 扬揚 扰擾 抚撫 抛拋 抠摳 抡掄 抢搶 护護 "
    "报報 担擔 拟擬 拢攏 拣揀 拥擁 拦攔 拧擰 拨撥 择擇 挂掛 挚摯 挛攣 挝撾 挞撻 挟挾 挠撓 挡擋 挣掙 挤擠 挥揮 捞撈 损損 捡撿 换換 捣搗 据據 掳擄 掷擲 掸撣 掺摻 "
    "揽攬 搀攙 搁擱 搂摟 搅攪 携攜 摄攝 摆擺 摇搖 摈擯 摊攤 撑撐 撵攆 擞擻 攒攢 敌敵 敛斂 数數 斋齋 斗鬥 斩斬 断斷 无無 旧舊 时時 旷曠 昼晝 显顯 晋晉 晒曬 晓曉 "
    "晕暈 暂暫 术術 机機 杀殺 杂雜 权權 杠槓 条條 来來 杨楊 杰傑 极極 构構 枢樞 枣棗 枪槍 枫楓 柜櫃 柠檸 栅柵 标標 栈棧 栋棟 栏欄 树樹 栖棲 样樣 档檔 桥橋 桨槳 "
    "桩樁 梦夢 检檢 棱稜 椭橢 楼樓 槛檻 横橫 樱櫻 橱櫥 欢歡 欧歐 歼殲 残殘 殴毆 毁毀 毕畢 毙斃 毡氈 气氣 氢氫 汇匯 汉漢 汤湯 汹洶 沟溝 没沒 沤漚 沥瀝 沦淪 沧滄 "
    "沪滬 泞濘 泪淚 泻瀉 泼潑 泽澤 洁潔 洒灑 洼窪 浅淺 浆漿 浇澆 浊濁 测測 济濟 浑渾 浓濃 涂塗 涌湧 涛濤 涝澇 涟漣 涡渦 涣渙 涤滌 润潤 涧澗 涨漲 涩澀 淀澱 渊淵 "
    "渍漬 渐漸 渔漁 渗滲 温溫 湾灣 湿濕 溃潰 溅濺 滚滾 滞滯 满滿 滤濾 滥濫 滦灤 滨濱 滩灘 漓灕 潍濰 潜潛 澜瀾 濒瀕 灭滅 灯燈 灵靈 灾災 灿燦 炉爐 点點 炼煉 炽熾 "
    "烁爍 烂爛 烃烴 烛燭 烟煙 烦煩 烧燒 烩燴 烫燙 烬燼 热熱 焕煥 爱愛 爷爺 牵牽 牺犧 犊犢 状狀 犹猶 狈狽 狞獰 独獨 狭狹 狮獅 狰猙 狱獄 猎獵 猪豬 猫貓 献獻 獭獺 "
    "玛瑪 环環 现現 珐琺 琐瑣 琼瓊 瑶瑤 瓮甕 电電 画畫 畅暢 畴疇 疗療 疟瘧 疡瘍 疮瘡 疯瘋 痈癰 痉痙 痒癢 痪瘓 瘪癟 瘫癱 癣癬 皑皚 皱皺 盏盞 盐鹽 监監 盖蓋 盗盜 "
    "盘盤 着著 睁睜 瞒瞞 瞩矚 矫矯 矾礬 矿礦 码碼 砖磚 砚硯 砾礫 础礎 硕碩 确確 硷鹼 碍礙 碱鹼 礼禮 祷禱 祸禍 禄祿 离離 秃禿 种種 积積 称稱 秽穢 税稅 稳穩 穷窮 "
    "窃竊 窍竅 窑窯 窜竄 窝窩 窥窺 竖豎 竞競 笋筍 笔筆 笺箋 笼籠 筑築 筛篩 筹籌 签簽 简簡 箩籮 篓簍 篮籃 篱籬 类類 粤粵 粪糞 粮糧 紧緊 纠糾 红紅 纤纖 约約 级級 "
    "纪紀 纫紉 纬緯 纯純 纱紗 纲綱 纳納 纵縱 纶綸 纷紛 纸紙 纹紋 纺紡 纽紐 线線 练練 组組 绅紳 细細 织織 终終 绊絆 绍紹 绎繹 经經 绑綁 绒絨 结結 绕繞 绘繪 给給 "
    "绚絢 络絡 绝絕 绞絞 统統 绢絹 绣繡 绥綏 绦縧 继繼 绩績 绪緒 续續 绰綽 绳繩 维維 绵綿 绷繃 绸綢 综綜 绽綻 绿綠 缀綴 缄緘 缅緬 缆纜 缉緝 缎緞 缓緩 缔締 缕縷 "
    "编編 缘緣 缚縛 缝縫 缠纏 缨纓 缩縮 缮繕 缴繳 网網 罗羅 罚罰 罢罷 羡羨 翘翹 耸聳 耻恥 聂聶 聋聾 职職 联聯 聪聰 肃肅 肠腸 肤膚 肮骯 肾腎 肿腫 胀脹 胁脅 胆膽 "
    "胜勝 胶膠 脉脈 脐臍 脑腦 脓膿 脚腳 脱脫 脸臉 腻膩 腾騰 舆輿 舰艦 舱艙 艰艱 艳艷 艺藝 节節 芜蕪 芦蘆 苇葦 苍蒼 苏蘇 苹蘋 茎莖 茧繭 荆荊 荐薦 荚莢 荡蕩 荣榮 "
    "荤葷 荧熒 荫蔭 药藥 莱萊 莲蓮 获獲 莹瑩 萝蘿 萤螢 营營 萧蕭 萨薩 葱蔥 蒋蔣 蓝藍 蓟薊 蔷薔 蔼藹 蕴蘊 虏虜 虑慮 虚虛 虫蟲 虱蝨 虽雖 虾蝦 蚀蝕 蚁蟻 蚂螞 蚕蠶 "
    "蛊蠱 蛮蠻 蛰蟄 蜕蛻 蜗蝸 蜡蠟 蝇蠅 蝉蟬 蝎蠍 衅釁 衔銜 补補 衬襯 袄襖 袜襪 袭襲 装裝 裤褲 见見 观觀 规規 觅覓 视視 览覽 觉覺 触觸 誉譽 誊謄 计計 订訂 讣訃 "
    "认認 讥譏 讨討 让讓 讫訖 训訓 议議 讯訊 记記 讲講 讳諱 讶訝 许許 讹訛 论論 讼訟 讽諷 设設 访訪 诀訣 证證 评評 诅詛 识識 诈詐 诉訴 诊診 诌謅 词詞 译譯 试試 "
    "诗詩 诚誠 诛誅 话話 诞誕 诡詭 询詢 诣詣 该該 详詳 诧詫 诫誡 诬誣 语語 误誤 诱誘 诲誨 说說 诵誦 请請 诸諸 诺諾 读讀 诽誹 课課 谁誰 调調 谅諒 谆諄 谈談 谊誼 "
    "谋謀 谍諜 谎謊 谐諧 谓謂 谗讒 谚諺 谜謎 谢謝 谣謠 谤謗 谦謙 谨謹 谩謾 谬謬 谭譚 谰讕 谱譜 谴譴 贝貝 贞貞 负負 贡貢 财財 责責 贤賢 败敗 账帳 货貨 质質 贩販 "
    "贪貪 贫貧 贬貶 购購 贮貯 贯貫 贰貳 贱賤 贴貼 贵貴 贷貸 贸貿 费費 贺賀 贼賊 贾賈 贿賄 赁賃 赂賂 赃贓 资資 赊賒 赋賦 赌賭 赎贖 赏賞 赐賜 赔賠 赖賴 赘贅 赚賺 "
    "赛賽 赞贊 赠贈 赡贍 赢贏 赣贛 赵趙 赶趕 趋趨 跃躍 践踐 踊踴 踌躊 踪蹤 蹿躥 躯軀 车車 轧軋 轨軌 轩軒 转轉 轮輪 软軟 轰轟 轴軸 轻輕 载載 轿轎 较較 辅輔 辆輛 "
    "辈輩 辉輝 辊輥 辐輻 辑輯 输輸 辕轅 辖轄 辗輾 辙轍 辞辭 辩辯 辫辮 边邊 辽遼 达達 迁遷 过過 迈邁 运運 还還 这這 进進 远遠 违違 连連 迟遲 迹跡 适適 选選 逊遜 "
    "递遞 逻邏 遗遺 遥遙 邓鄧 邮郵 邹鄒 邻鄰 郑鄭 郧鄖 郸鄲 酝醖 酱醬 酿釀 释釋 鉴鑒 针針 钉釘 钎釺 钒釩 钓釣 钙鈣 钝鈍 钞鈔 钟鐘 钠鈉 钡鋇 钢鋼 钥鑰 钦欽 钧鈞 "
    "钨鎢 钩鉤 钮鈕 钱錢 钳鉗 钵鉢 钻鑽 钾鉀 铀鈾 铁鐵 铂鉑 铃鈴 铅鉛 铆鉚 铜銅 铝鋁 铡鍘 铣銑 铬鉻 铭銘 铰鉸 铱銥 铲鏟 银銀 铸鑄 铺鋪 链鏈 销銷 锁鎖 锄鋤 锅鍋 "
    "锈鏽 锋鋒 锌鋅 锐銳 锑銻 锗鍺 错錯 锚錨 锡錫 锣鑼 锤錘 锥錐 锦錦 锨鍁 锭錠 键鍵 锯鋸 锰錳 锹鍬 锻鍛 镀鍍 镁鎂 镇鎮 镊鑷 镍鎳 镐鎬 镑鎊 镜鏡 镣鐐 镭鐳 镰鐮 "
    "镶鑲 长長 门門 闪閃 闭閉 问問 闯闖 闰閏 闲閒 间間 闷悶 闸閘 闹鬧 闺閨 闻聞 闽閩 阀閥 阁閣 阂閡 阅閱 阉閹 阎閻 阐闡 阑闌 阔闊 队隊 阳陽 阴陰 阵陣 阶階 际際 "
    "陆陸 陇隴 陈陳 陕陝 陨隕 险險 随隨 隐隱 隶隸 难難 雏雛 雾霧 静靜 韦韋 韧韌 韩韓 韵韻 页頁 顶頂 顷頃 项項 顺順 须須 顽頑 顾顧 顿頓 颁頒 颂頌 预預 颅顱 领領 "
    "颇頗 颈頸 颊頰 颐頤 频頻 颓頹 颖穎 颗顆 题題 颜顏 额額 颠顛 颤顫 颧顴 风風 飘飄 飞飛 饥飢 饭飯 饮飲 饯餞 饰飾 饱飽 饲飼 饵餌 饶饒 饺餃 饼餅 饿餓 馁餒 馅餡 "
    "馆館 馈饋 馋饞 馏餾 馒饅 马馬 驭馭 驮馱 驯馴 驰馳 驱驅 驳駁 驴驢 驶駛 驹駒 驻駐 驼駝 驾駕 骂罵 骄驕 骆駱 骇駭 骋騁 验驗 骏駿 骑騎 骗騙 骚騷 骡騾 骤驟 鱼魚 "
    "鲁魯 鲍鮑 鲜鮮 鲤鯉 鲸鯨 鳃鰓 鳖鱉 鳞鱗 鸟鳥 鸡雞 鸣鳴 鸥鷗 鸦鴉 鸭鴨 鸯鴦 鸳鴛 鸵鴕 鸽鴿 鸿鴻 鹃鵑 鹅鵝 鹊鵲 鹏鵬 鹤鶴 鹰鷹 麦麥 黄黃 齐齊 齿齒 龄齡 龋齲 "
    "龙龍 龚龔 龟龜 谥諡"
)
_TEXT_FIX = {"钟": "鐘"}  # (a surname above: 鍾; the word: 鐘)
TEXT_S2T: dict[str, str] = {
    **S2T,
    **{pair[0]: pair[1] for pair in _TEXT_PAIRS.split()},
    **_TEXT_FIX,
}
# words whose traditional form is not the character-by-character one (or must stay as they are)
_TEXT_WORDS = {
    "头发": "頭髮",
    "理发": "理髮",
    "发型": "髮型",
    "皇后": "皇后",
    "太后": "太后",
    "干净": "乾淨",
    "干杯": "乾杯",
    "干燥": "乾燥",
    "干嘛": "幹嘛",
    "面条": "麵條",
    "面包": "麵包",
    "面粉": "麵粉",
    "日历": "日曆",
    "农历": "農曆",
    "周末": "週末",
    "这周": "這週",
    "本周": "本週",
    "上周": "上週",
    "下周": "下週",
    "每周": "每週",
    "一只": "一隻",
    "两只": "兩隻",
    "三只": "三隻",
    "这只": "這隻",
    "那只": "那隻",
}
# 里 is 裡 inside (这里 家里 手里 里面 里头 …), and 里 as in 公里 or 邻里 otherwise
_LI = "(?P<li>(?<=[这那哪家屋心手眼嘴间子])里|里(?=[面头边外]))"
_TEXT_TOKEN = re.compile(
    "|".join([*(re.escape(w) for w in sorted(_TEXT_WORDS, key=len, reverse=True)), _LI, "."]),
    re.DOTALL,
)
_OTHER_SCRIPT = re.compile("[\u3040-\u30ff\uac00-\ud7af]")  # kana, hangul: not Chinese text


def to_traditional_text(text: str) -> str:
    """``text`` in traditional characters (see the section above). English, numbers, punctuation
    and what is already traditional are untouched; a text with kana or hangul in it is left as it
    is (Japanese uses its own forms of the same characters)."""
    if not text or _OTHER_SCRIPT.search(text):
        return text

    def one(match: re.Match[str]) -> str:
        token = match.group(0)
        if match.lastgroup == "li":
            return "裡"
        word = _TEXT_WORDS.get(token)
        return word if word is not None else TEXT_S2T.get(token, token)

    return _TEXT_TOKEN.sub(one, text)


# fillers the model sometimes leaves in front of the name
_LEAD = ("這位是", "这位是", "這是", "这是", "他是", "她是", "我的", "叫做", "叫", "名字是")

# words that are not names: who the family points at, not what they call them
GENERIC_PERSON = frozenset(
    {
        "人",
        "他",
        "她",
        "你",
        "我",
        "大家",
        "家人",
        "客人",
        "朋友",
        "小朋友",
        "孩子",
        "小孩",
        "說話的人",
        "那個人",
        "這個人",
        "someone",
        "person",
        "people",
        "he",
        "she",
        "they",
        "who",
        "speaker",
    }
)

# pinyin (initial, final, tone) for the characters of the names seen and of family words: enough
# to tell "dan huang" from "zheng hua" by ear; anything not here compares by character only
PINYIN: dict[str, tuple[str, str, int]] = {
    "蛋": ("d", "an", 4),
    "黃": ("h", "uang", 2),
    "鄭": ("zh", "eng", 4),
    "華": ("h", "ua", 2),
    "媽": ("m", "a", 1),
    "爸": ("b", "a", 4),
    "爺": ("", "ie", 2),
    "奶": ("n", "ai", 3),
    "姐": ("j", "ie", 3),
    "哥": ("g", "e", 1),
    "弟": ("d", "i", 4),
    "妹": ("m", "ei", 4),
    "叔": ("sh", "u", 1),
    "姨": ("", "i", 2),
    "舅": ("j", "iu", 4),
    "婆": ("p", "o", 2),
    "公": ("g", "ong", 1),
    "伯": ("b", "o", 2),
    "姑": ("g", "u", 1),
    "寶": ("b", "ao", 3),
    "貝": ("b", "ei", 4),
    "小": ("x", "iao", 3),
    "美": ("m", "ei", 3),
    "明": ("m", "ing", 2),
    "玲": ("l", "ing", 2),
    "芳": ("f", "ang", 1),
    "偉": ("w", "ei", 3),
    "強": ("q", "iang", 2),
    "麗": ("l", "i", 4),
    "靜": ("j", "ing", 4),
    "軍": ("j", "un", 1),
    "傑": ("j", "ie", 2),
    "磊": ("l", "ei", 3),
    "王": ("w", "ang", 2),
    "李": ("l", "i", 3),
    "張": ("zh", "ang", 1),
    "劉": ("l", "iu", 2),
    "陳": ("ch", "en", 2),
    "楊": ("y", "ang", 2),
    "吳": ("w", "u", 2),
    "趙": ("zh", "ao", 4),
    "周": ("zh", "ou", 1),
    "林": ("l", "in", 2),
    "蔡": ("c", "ai", 4),
    "旺": ("w", "ang", 4),
    "財": ("c", "ai", 2),
    "福": ("f", "u", 2),
    "樂": ("l", "e", 4),
    "豆": ("d", "ou", 4),
    "團": ("t", "uan", 2),
    "圓": ("y", "uan", 2),
}
_INITIAL_CLASS = {
    **{c: 0 for c in ("b", "p", "m", "f")},
    **{c: 1 for c in ("d", "t", "n", "l")},
    **{c: 2 for c in ("g", "k", "h")},
    **{c: 3 for c in ("j", "q", "x")},
    **{c: 4 for c in ("zh", "ch", "sh", "r")},
    **{c: 5 for c in ("z", "c", "s")},
    **{c: 6 for c in ("y", "w", "")},
}
# (unaspirated d/z/zh, j and b/g are the near ones of a listener's ear: a mishearing keeps the
# manner and shifts the place)
_NEAR_INITIALS = frozenset(
    {("d", "z"), ("d", "zh"), ("t", "c"), ("t", "ch"), ("l", "n"), ("sh", "s"), ("zh", "z")}
)
_VOWELS = "aeiouü"


def to_traditional(text: str) -> str:
    return "".join(S2T.get(c, c) for c in text)


def _is_drop(c: str) -> bool:
    category = unicodedata.category(c)
    return category[0] in ("Z", "P", "S", "C") or c in "·・"


def normalise_name(raw: str) -> str:
    """One written form for a name (see the module doc). ``""`` when nothing of a name is left."""
    text = unicodedata.normalize("NFKC", str(raw))
    text = "".join(c for c in text if not _is_drop(c))
    text = to_traditional(text)
    for lead in _LEAD:
        lead_t = to_traditional(lead)
        if text.startswith(lead_t) and len(text) > len(lead_t):
            text = text[len(lead_t) :]
            break
    return text


def vocabulary(duck_name: str, known: Iterable[str]) -> tuple[str, ...]:
    """The words the transcriber is told to listen for (the SDK's ``custom_vocabulary``): the
    duck's own name first, then the names the family has given (the learner's known people), each
    once (the same name written another way counts as the same), at most :data:`MAX_VOCABULARY`.
    A name is data: it goes as written, never translated."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (duck_name, *known):
        text = " ".join(str(raw or "").split())
        if not text or len(text) > MAX_LATIN_NAME:
            continue
        marker = key(text)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        out.append(text)
        if len(out) >= MAX_VOCABULARY:
            break
    return tuple(out)


def key(name: str) -> str:
    """What two spellings of a name are compared by (case and width do not matter)."""
    return normalise_name(name).casefold()


def is_generic(text: str) -> bool:
    return normalise_name(text).casefold() in {normalise_name(w).casefold() for w in GENERIC_PERSON}


# ── how alike two names are ───────────────────────────────────────────────────────────


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        row = [i]
        for j, cb in enumerate(b, start=1):
            row.append(min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = row
    return previous[-1]


def _initial_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if (a, b) in _NEAR_INITIALS or (b, a) in _NEAR_INITIALS:
        return 0.6
    return 0.3 if _INITIAL_CLASS.get(a, -1) == _INITIAL_CLASS.get(b, -2) else 0.0


def _final_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    core_a, core_b = a.rstrip("ng"), b.rstrip("ng")
    if core_a == core_b:
        return 0.7  # (an / ang, in / ing: the nasal ending is the easiest to mishear)
    vowels_a = {c for c in a if c in _VOWELS}
    vowels_b = {c for c in b if c in _VOWELS}
    shared = len(vowels_a & vowels_b) / max(1, len(vowels_a | vowels_b))
    nasal = (a.endswith(("n", "ng"))) == (b.endswith(("n", "ng")))
    return 0.15 + 0.35 * shared + (0.15 if nasal else 0.0)


def _syllable_sim(a: tuple[str, str, int], b: tuple[str, str, int]) -> float:
    return 0.4 * _initial_sim(a[0], b[0]) + 0.4 * _final_sim(a[1], b[1]) + 0.2 * (a[2] == b[2])


def _sound_sim(a: str, b: str) -> float:
    """By ear: syllable by syllable for names made of characters the table knows (0 otherwise)."""
    pa = [PINYIN.get(c) for c in a]
    pb = [PINYIN.get(c) for c in b]
    if not a or not b or any(p is None for p in pa) or any(p is None for p in pb):
        return 0.0
    if abs(len(pa) - len(pb)) > 1:
        return 0.0
    n = min(len(pa), len(pb))
    scores = [_syllable_sim(pa[i], pb[i]) for i in range(n)]  # type: ignore[arg-type]
    return sum(scores) / max(len(pa), len(pb))


def similarity(a: str, b: str) -> float:
    """0..1: how alike two names are, by characters and by sound. 1 is the same name."""
    ka, kb = key(a), key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    chars = 1.0 - _levenshtein(ka, kb) / max(len(ka), len(kb))
    if (ka in kb or kb in ka) and min(len(ka), len(kb)) >= 1:
        chars = max(chars, 0.8)  # (媽 / 媽媽: one dropped or doubled character)
    return max(0.0, max(chars, _sound_sim(ka, kb)))


# ── what to do with a proposed name ───────────────────────────────────────────────────

BIND = "bind"
PENDING = "pending"
REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class NameDecision:
    action: str  # BIND | PENDING | REFUSE
    reason: str
    name: str  # normalised ("" if nothing was left)
    pid: int | None = None  # the person (of the duck's own ids) it applies to
    score: float = 0.0  # the best similarity found (with the duck's own name or a known one)
    near: str | None = None  # … and what it sounded like


def decide_name(
    raw: str,
    *,
    in_view_pid: int | None,
    in_view_name: str | None,
    known: Mapping[int, str],
    duck_name: str,
    speech_age_s: float | None,
) -> NameDecision:
    """What to do with ``raw``, a name proposed for the person the duck sees right now.

    ``in_view_pid`` / ``in_view_name``: the person the duck itself perceives in view (its own id)
    and the name it already has for them; ``known``: the family-name list (person id → name);
    ``duck_name``: its own; ``speech_age_s``: seconds since a person last spoke (None: nobody has).
    """
    name = normalise_name(raw)
    if not name:
        return NameDecision(REFUSE, "name_empty", "")
    limit = MAX_LATIN_NAME if name.isascii() else MAX_NAME_CHARS
    if len(name) > limit:
        return NameDecision(REFUSE, "name_too_long", name)
    if name.isdigit() or is_generic(name):
        return NameDecision(REFUSE, "not_a_name", name)
    own = similarity(name, duck_name) if duck_name else 0.0
    if duck_name and key(name) == key(duck_name):
        return NameDecision(REFUSE, "duck_name", name, score=1.0, near=duck_name)
    if own >= SIM_OWN_NAME:
        return NameDecision(REFUSE, "sounds_like_duck_name", name, score=own, near=duck_name)
    if in_view_pid is None:
        return NameDecision(REFUSE, "no_person_in_view", name)
    best_pid, best = None, 0.0
    for pid, other in known.items():
        score = similarity(name, other)
        if score > best:
            best_pid, best = pid, score
    if best_pid is not None and best >= SIM_KNOWN_NAME:
        other = known[best_pid]
        if best_pid == in_view_pid:
            return NameDecision(BIND, "confirmed", other, in_view_pid, best, other)
        # the name of someone else it knows, said while this one is in view: it may be that one
        # seen wrongly, or two people with the same name — not for a name to settle
        return NameDecision(PENDING, "name_taken", name, in_view_pid, best, other)
    if in_view_name is not None and key(in_view_name) != key(name):
        return NameDecision(PENDING, "conflict", name, in_view_pid, best, in_view_name)
    if speech_age_s is None or speech_age_s > RECENT_SPEECH_S:
        return NameDecision(PENDING, "no_recent_speech", name, in_view_pid, best)
    return NameDecision(BIND, "new_name", name, in_view_pid, best)


# a pending name may become the name after this many separate pieces of evidence — but only
# when the doubt was about *when* or *whether to rename*, never about whose name it is
PROMOTABLE = frozenset({"conflict", "no_recent_speech"})


def promotable(reason: str, evidence: int) -> bool:
    return reason in PROMOTABLE and evidence >= PROMOTE_AT


# ── a hint that names a person ────────────────────────────────────────────────────────


def match_known(hint: str, known: Mapping[int, str]) -> int | None:
    """The person a hint ("媽媽", "妈妈", "我的媽媽") names among the family it knows, or None."""
    best_pid, best = None, 0.0
    for pid, other in known.items():
        score = similarity(hint, other)
        if score > best:
            best_pid, best = pid, score
    return best_pid if best >= SIM_KNOWN_NAME else None


__all__ = [
    "BIND",
    "GENERIC_PERSON",
    "MAX_NAME_CHARS",
    "PENDING",
    "PROMOTABLE",
    "PROMOTE_AT",
    "REFUSE",
    "NameDecision",
    "decide_name",
    "is_generic",
    "key",
    "match_known",
    "normalise_name",
    "promotable",
    "similarity",
    "to_traditional",
    "to_traditional_text",
]
