"""The words people say to the duck: the one table the text parser reads.

``PHRASES[verb][language]`` is the list of phrases that mean that verb in that language
(``zh-Hant`` 繁體中文, ``en`` English, ``ja`` 日本語, ``es`` Español, ``zh-Hans`` 简体中文, ``ko`` 한국어).
Nothing else in the program knows a command word: add or fix a phrase here and the parser, the
tests and the page's hints follow.

Rules the parser (``duckmind.neuro.commands``) applies to every phrase:

* The text is folded to NFKC, lower case, Latin accents dropped (``siéntate`` = ``sientate``), and
  the duck's name is taken out of it. Write the phrases the same way: lower case, no Latin accents.
* A phrase matches inside the sentence, so ``please sit down for me`` and 「蛋黃請坐下」 work without
  a polite-form list. A phrase made only of Latin letters and spaces must stand on word borders
  (``ven`` is not found inside ``vengo``). CJK and Hangul phrases match as plain text.
* A phrase that starts with ``=`` is too short or too common to be found inside a sentence
  (停, ``down``, ``para``, 자): it counts only when it is the whole utterance (the name, the
  punctuation and the small filler words below left out).
* The longest matching phrase wins; a tie goes to the one said first. A negation just before the
  match (「不要坐下」, ``don't sit``, ``no te sientes``) means the sentence is not that command.

``zh-Hant`` and ``zh-Hans`` share many phrases (坐下); such a phrase belongs to both languages and
the language of the command is then the viewer's own choice (``lang`` of the request), else
``zh-Hant``, the source language.
"""

from __future__ import annotations

LANGS: tuple[str, ...] = ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko")

# The built-in verbs (every duck does them when awake, not afraid, not held, not mid-flight) ...
BUILTIN: tuple[str, ...] = ("sit", "stand", "come", "look_at_me", "stop", "follow", "go_home")
# ... the tricks (the duck has to be taught them; in phase 1 they are locked: it hears, it does not know)
TRICKS: tuple[str, ...] = ("lie_down", "spin", "bow", "shake", "flip")
# A verb the page makes by pointing at the floor, never by words.
POINTED: tuple[str, ...] = ("go_there",)

PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "sit": {
        "zh-Hant": ("坐下", "坐好", "坐一下", "請坐", "坐著", "=坐"),
        "en": ("sit", "sit down", "take a seat", "have a seat", "sit please"),
        "ja": ("座って", "すわって", "おすわり", "お座り", "座れ", "座ってください"),
        "es": ("sientate", "sentado", "sentada", "sientese", "a sentarse", "sientate por favor"),
        "zh-Hans": ("坐下", "坐好", "坐一下", "请坐", "坐着", "=坐"),
        "ko": ("앉아", "앉아라", "앉으세요", "앉아줘", "앉아 주세요", "앉아봐"),
    },
    "stand": {
        "zh-Hant": ("站起來", "站好", "起立", "站立", "站著", "=起來"),
        "en": ("stand", "stand up", "get up", "up you go", "rise and stand"),
        "ja": ("立って", "立ち上がって", "立て", "たって", "立ってください", "立ちなさい"),
        "es": ("levantate", "ponte de pie", "de pie", "ponte en pie", "levanta"),
        "zh-Hans": ("站起来", "站好", "起立", "站立", "站着", "=起来"),
        "ko": ("일어나", "일어서", "일어나라", "일어서세요", "일어나봐", "=서"),
    },
    "come": {
        "zh-Hant": ("過來", "來這裡", "來這邊", "來我這", "到我這", "靠過來", "快過來", "=來"),
        "en": ("come", "come here", "come to me", "come over", "over here", "come on over"),
        "ja": ("おいで", "こっちにおいで", "こっち来て", "こっちに来て", "来て", "来い", "こっちへ来て"),
        "es": ("ven", "ven aqui", "ven aca", "acercate", "ven a mi", "ven para aca"),
        "zh-Hans": ("过来", "来这里", "来这边", "来我这", "到我这", "靠过来", "快过来", "=来"),
        "ko": ("이리 와", "이리와", "와라", "이리 오세요", "이쪽으로 와", "=와", "오세요"),
    },
    "look_at_me": {
        "zh-Hant": ("看我", "看著我", "看這裡", "看過來", "看這邊", "望著我"),
        "en": ("look at me", "look here", "watch me", "eyes on me", "look over here"),
        "ja": ("見て", "私を見て", "こっち見て", "こっちを見て", "僕を見て", "見ててください"),
        "es": ("mirame", "mira aqui", "mira aca", "mirame a mi", "mira", "mira para aca"),
        "zh-Hans": ("看我", "看着我", "看这里", "看过来", "看这边", "望着我"),
        "ko": ("나 봐", "날 봐", "나를 봐", "여기 봐", "쳐다봐", "나 좀 봐", "이쪽 봐", "=봐"),
    },
    "stop": {
        "zh-Hant": ("停下", "停下來", "停住", "停止", "不要動", "別動", "站住", "等一下", "=停"),
        "en": ("stop", "halt", "freeze", "stay", "stay there", "hold on", "wait", "stay still",
               "stand still", "don't move"),
        "ja": ("止まれ", "止まって", "ストップ", "待て", "待って", "動かないで", "じっとして"),
        "es": ("detente", "quieto", "quieta", "espera", "no te muevas", "quedate", "=para", "=alto"),
        "zh-Hans": ("停下", "停下来", "停住", "停止", "不要动", "别动", "站住", "等一下", "=停"),
        "ko": ("멈춰", "정지", "그만", "기다려", "움직이지 마", "가만히", "스톱", "멈추세요"),
    },
    "follow": {
        "zh-Hant": ("跟著我", "跟我來", "跟我走", "跟上", "跟隨", "跟過來", "=跟著"),
        "en": ("follow me", "follow", "come with me", "walk with me", "heel"),
        "ja": ("ついてきて", "付いてきて", "ついておいで", "ついて来て", "ついてこい", "一緒に来て"),
        "es": ("sigueme", "sigame", "ven conmigo", "acompaname", "ven detras de mi"),
        "zh-Hans": ("跟着我", "跟我来", "跟我走", "跟上", "跟随", "跟过来", "=跟着"),
        "ko": ("따라와", "따라 와", "따라오세요", "나를 따라와", "따라와라", "같이 가자"),
    },
    "go_home": {
        "zh-Hant": ("回家", "回去", "回窩", "回巢", "去充電", "回充電", "回你的窩"),
        "en": ("go home", "go to your nest", "go to your dock", "go charge", "back to your nest",
               "=home"),
        "ja": ("家に帰って", "お家に帰って", "おうちに帰って", "帰って", "帰れ", "巣に戻って",
               "充電して", "ハウス"),
        "es": ("ve a casa", "vete a casa", "a casa", "ve a tu nido", "vuelve a casa",
               "ve a cargar", "vuelve al nido", "regresa a casa"),
        "zh-Hans": ("回家", "回去", "回窝", "回巢", "去充电", "回充电", "回你的窝"),
        "ko": ("집에 가", "집에 가자", "집으로 가", "집으로", "돌아가", "집에 가라",
               "집에 가세요", "충전하러 가"),
    },
    "lie_down": {
        "zh-Hant": ("趴下", "趴著", "躺下", "睡午覺", "睡覺", "午睡", "裝死", "趴好", "小睡"),
        "en": ("lie down", "lay down", "take a nap", "nap", "go to sleep", "sleep", "play dead",
               "nap time", "=down"),
        "ja": ("伏せ", "伏せて", "寝て", "お昼寝", "ねんね", "おねんね", "死んだふり", "昼寝して"),
        "es": ("echate", "tumbate", "acuestate", "duerme", "a dormir", "siesta", "haz la siesta",
               "hazte el muerto", "haz el muerto"),
        "zh-Hans": ("趴下", "趴着", "躺下", "睡午觉", "睡觉", "午睡", "装死", "趴好", "小睡"),
        "ko": ("엎드려", "누워", "낮잠", "낮잠 자", "죽은 척", "잠자", "자라", "누우세요", "=자"),
    },
    "spin": {
        "zh-Hant": ("轉圈", "轉圈圈", "轉一圈", "轉個圈", "轉身", "打轉"),
        "en": ("spin", "turn around", "twirl", "spin around", "do a spin", "=turn"),
        "ja": ("回って", "くるくる", "くるっと回って", "ターン", "回れ", "一回転して", "くるっと"),
        "es": ("gira", "da una vuelta", "da vueltas", "girar", "vuelta", "date la vuelta",
               "girate", "dar una vuelta"),
        "zh-Hans": ("转圈", "转圈圈", "转一圈", "转个圈", "转身", "打转"),
        "ko": ("빙글빙글", "한 바퀴", "돌아봐", "뱅글뱅글", "돌아라", "=돌아"),
    },
    "bow": {
        "zh-Hant": ("鞠躬", "敬禮", "行禮", "彎腰"),
        "en": ("bow", "take a bow", "curtsy", "salute"),
        "ja": ("お辞儀", "おじぎ", "敬礼", "お辞儀して", "おじぎして", "=礼"),
        "es": ("reverencia", "haz una reverencia", "inclinate", "saluda"),
        "zh-Hans": ("鞠躬", "敬礼", "行礼", "弯腰"),
        "ko": ("인사", "인사해", "절해", "고개 숙여", "인사해봐", "=절"),
    },
    "shake": {
        "zh-Hant": ("甩一甩", "甩甩", "甩甩身體", "抖一抖", "抖抖", "甩一下", "甩毛"),
        "en": ("shake", "shake it off", "shake off", "shake yourself", "give a shake",
               "shake your body"),
        "ja": ("ぶるぶる", "ブルブル", "振るって", "身震い", "体を振って", "身体を振って"),
        "es": ("sacudete", "sacude", "sacudete el cuerpo", "agitate"),
        "zh-Hans": ("甩一甩", "甩甩", "甩甩身体", "抖一抖", "抖抖", "甩一下", "甩毛"),
        "ko": ("털어", "털어봐", "몸 털어", "흔들어", "부르르", "몸을 털어"),
    },
    "flip": {
        "zh-Hant": ("翻跟斗", "翻筋斗", "翻滾", "翻個跟斗", "前滾翻", "翻一個", "翻轉"),
        "en": ("flip", "do a flip", "somersault", "roll over", "roll", "backflip", "front flip",
               "forward roll", "tumble"),
        "ja": ("宙返り", "ちゅうがえり", "前転", "前転して", "でんぐり返し", "でんぐり返しして",
               "ひっくり返って"),
        "es": ("voltereta", "da una voltereta", "haz una voltereta", "vuelta de campana", "rueda",
               "rueda hacia adelante", "haz un flip", "flip"),
        "zh-Hans": ("翻跟头", "翻筋斗", "翻滚", "翻个跟头", "前滚翻", "翻跟斗", "翻转"),
        "ko": ("구르기", "앞구르기", "구르세요", "구르라", "굴러", "공중제비", "재주넘기", "텀블링"),
    },
}

# Words that mean "no" when they stand just before a command phrase: the sentence is then not that
# command (「不要坐下」, "don't sit"). Japanese and Korean put the "no" after the verb, so a phrase
# in them is never found inside a negated sentence to begin with.
NEGATIONS_BEFORE: dict[str, tuple[str, ...]] = {
    "zh": ("不要", "別", "别", "不用", "勿", "不可以", "不能", "不許", "不准", "不"),
    "en": ("don't", "dont", "do not", "never", "no", "not"),
    "es": ("no", "nunca"),
}

# Small words left out before a whole-utterance (``=``) phrase is compared: politeness, particles.
FILLERS: tuple[str, ...] = (
    "請", "请", "麻煩你", "麻烦你", "拜託", "拜托", "啦", "吧", "喔", "哦", "啊", "呀", "嘛", "了",
    "please", "pls", "now", "right now", "por favor", "ya", "ahora", "ください", "ね", "よ", "!",
    "부탁해", "좀", "제발", "요",
)
