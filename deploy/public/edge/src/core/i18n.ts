/**
 * The edge pages' words in six languages. 繁體中文 is the source; the others follow the project's
 * glossary (src/duckmind/i18n/glossary.json: 小鴨 → the duck / アヒル / el patito / 小鸭 / 오리,
 * "Pollen's", never "official"). Values in braces are filled in by `t()`; every language uses the
 * same values (test/i18n.test.ts checks).
 */

export const LANGS = ["zh-Hant", "en", "ja", "es", "zh-Hans", "ko"] as const;
export type Lang = (typeof LANGS)[number];

/** Language names are shown as they are, in their own script (data, not translated). */
export const LANG_NAMES: Record<Lang, string> = {
	"zh-Hant": "繁體中文",
	en: "English",
	ja: "日本語",
	es: "Español",
	"zh-Hans": "简体中文",
	ko: "한국어",
};

type Catalog = Record<string, string>;

const zhHant: Catalog = {
	"page.title": "小鴨模擬器・公開試玩",
	"page.brand": "小鴨模擬器",
	"page.lang": "語言",
	"land.eyebrow": "公開試玩",
	"land.h1a": "養一隻小鴨",
	"land.h1b": "{minutes} 分鐘",
	"land.lead": "每位訪客會拿到一隻全新的隨機小鴨，在自己的模擬器裡玩 {minutes} 分鐘。時間到了，可以把牠下載成 .duck 檔帶走。",
	"land.how1": "完成真人驗證，按下按鈕。",
	"land.how2": "有空位就馬上開始；沒有的話就排隊，這頁會告訴你排第幾。",
	"land.how3": "小鴨醒來要一點時間，大約一兩分鐘。",
	"land.stats": "現在有 {live} 隻小鴨在玩、{queued} 人在排隊（最多同時 {max} 隻）。",
	"land.start": "給我一隻小鴨",
	"land.verify": "先完成上面的驗證，再按按鈕。",
	"land.nc.title": "非商業用途",
	"land.nc.text": "這個試玩免費、沒有廣告、不收費。小鴨的 3D 模型只能用在非商業用途。",
	"land.attr": "小鴨的 3D 模型來自 Pollen Robotics 的 Microduck，授權是 CC BY-NC-SA 4.0。duckmind 是非官方專案，與 Pollen Robotics 無關。",
	"land.license": "授權條款",
	"land.privacy": "我們只用一個 cookie 認出你的試玩；你的網路位址只以雜湊的形式暫存，用來限制次數。",
	"err.turnstile": "驗證沒有通過，請再試一次。",
	"err.already": "你已經有一隻小鴨在玩或在排隊了。",
	"err.rate": "你剛剛已經開過好幾隻小鴨了，請晚一點再來。",
	"err.busy": "現在排隊的人太多了，請晚一點再來。",
	"err.start": "小鴨沒有醒來（伺服器出了問題），請再試一次。",
	"queue.title": "排隊中",
	"queue.place": "你排第 {place} 位",
	"queue.note": "每隻小鴨玩 {minutes} 分鐘。輪到你時，這頁會自己打開小鴨。請不要關掉這頁：離開一分鐘就會失去位置。",
	"queue.leave": "離開隊伍",
	"start.title": "小鴨正在醒來",
	"start.note": "正在幫你準備一隻新的小鴨，通常要一兩分鐘。準備好了，這頁會自己打開。",
	"wait.offline": "連不上伺服器，正在重試…",
	"gone.title": "時間到了",
	"gone.note": "謝謝你來玩！這隻小鴨已經去休息了。",
	"gone.crash": "小鴨的模擬器意外停了，對不起。",
	"gone.start": "小鴨沒有醒來，對不起。",
	"gone.idle": "你離開了一陣子，小鴨先去休息了。",
	"gone.place": "你離開太久，失去了排隊的位置。",
	"gone.left": "你離開了，小鴨去休息了。",
	"gone.again": "再來一隻",
};

const en: Catalog = {
	"page.title": "Duck Simulator · public try-out",
	"page.brand": "Duck Simulator",
	"page.lang": "Language",
	"land.eyebrow": "Public try-out",
	"land.h1a": "Raise a duck",
	"land.h1b": "for {minutes} minutes",
	"land.lead": "Every visitor gets a brand-new random duck to play with in its own simulator for {minutes} minutes. When the time is up, you can download it as a .duck file and keep it.",
	"land.how1": "Pass the human check and press the button.",
	"land.how2": "If a spot is free you start right away; if not, you wait in line and this page shows your place.",
	"land.how3": "The duck takes a little while to wake up, about a minute or two.",
	"land.stats": "{live} ducks playing now, {queued} people waiting (at most {max} at once).",
	"land.start": "Get a duck",
	"land.verify": "Finish the check above first, then press the button.",
	"land.nc.title": "Non-commercial",
	"land.nc.text": "This try-out is free: no ads, no fees. The duck's 3D model may only be used for non-commercial purposes.",
	"land.attr": "The duck's 3D model is Pollen Robotics' Microduck, licensed CC BY-NC-SA 4.0. duckmind is an unofficial project, not affiliated with Pollen Robotics.",
	"land.license": "License",
	"land.privacy": "We use one cookie to recognise your try-out; your network address is kept only as a hash for a short time, to limit how often it can start.",
	"err.turnstile": "The check did not pass. Please try again.",
	"err.already": "You already have a duck playing or waiting in line.",
	"err.rate": "You have started several ducks just now. Please come back a bit later.",
	"err.busy": "Too many people are waiting right now. Please come back a bit later.",
	"err.start": "The duck did not wake up (a server problem). Please try again.",
	"queue.title": "Waiting in line",
	"queue.place": "You are number {place}",
	"queue.note": "Each duck plays for {minutes} minutes. When it is your turn, this page opens your duck by itself. Please keep this page open: away for a minute and you lose your place.",
	"queue.leave": "Leave the line",
	"start.title": "Your duck is waking up",
	"start.note": "Getting a new duck ready for you; this usually takes a minute or two. This page opens it by itself when it is ready.",
	"wait.offline": "Can't reach the server, trying again…",
	"gone.title": "Time is up",
	"gone.note": "Thanks for playing! This duck has gone to rest.",
	"gone.crash": "The duck's simulator stopped unexpectedly. Sorry.",
	"gone.start": "The duck did not wake up. Sorry.",
	"gone.idle": "You were away for a while, so the duck went to rest.",
	"gone.place": "You were away too long and lost your place in line.",
	"gone.left": "You left, and the duck went to rest.",
	"gone.again": "Get another duck",
};

const ja: Catalog = {
	"page.title": "アヒル・シミュレーター・公開おためし",
	"page.brand": "アヒル・シミュレーター",
	"page.lang": "言語",
	"land.eyebrow": "公開おためし",
	"land.h1a": "アヒルを育てよう",
	"land.h1b": "{minutes} 分間",
	"land.lead": "来た人ごとに、新しいランダムなアヒルが一羽。自分だけのシミュレーターで {minutes} 分間遊べます。時間になったら、.duck ファイルとしてダウンロードして持ち帰れます。",
	"land.how1": "人間の確認をすませて、ボタンを押します。",
	"land.how2": "空きがあればすぐ始まります。なければ順番待ちで、このページに何番目か出ます。",
	"land.how3": "アヒルが目をさますまで、一、二分ほどかかります。",
	"land.stats": "いま {live} 羽が遊んでいて、{queued} 人が待っています（同時に最大 {max} 羽）。",
	"land.start": "アヒルをもらう",
	"land.verify": "先に上の確認をすませてから、ボタンを押してください。",
	"land.nc.title": "非商用",
	"land.nc.text": "このおためしは無料で、広告も料金もありません。アヒルの 3D モデルは非商用でしか使えません。",
	"land.attr": "アヒルの 3D モデルは Pollen Robotics の Microduck で、ライセンスは CC BY-NC-SA 4.0 です。duckmind は非公式のプロジェクトで、Pollen Robotics とは関係ありません。",
	"land.license": "ライセンス",
	"land.privacy": "おためしを見分けるために cookie を一つだけ使います。ネットワークのアドレスは回数を制限するため、ハッシュの形で少しのあいだだけ保存します。",
	"err.turnstile": "確認が通りませんでした。もう一度どうぞ。",
	"err.already": "もう遊んでいるか、順番待ちのアヒルがいます。",
	"err.rate": "さっきから何羽も始めています。少し時間をおいてから来てください。",
	"err.busy": "いま待っている人が多すぎます。少し時間をおいてから来てください。",
	"err.start": "アヒルが目をさましませんでした（サーバーの問題）。もう一度どうぞ。",
	"queue.title": "順番待ち",
	"queue.place": "あなたは {place} 番目",
	"queue.note": "一羽 {minutes} 分間遊べます。順番が来たら、このページが自動でアヒルを開きます。このページは閉じないでください。一分はなれると順番がなくなります。",
	"queue.leave": "列をはなれる",
	"start.title": "アヒルが目をさましています",
	"start.note": "新しいアヒルを用意しています。たいてい一、二分です。準備ができたら、このページが自動で開きます。",
	"wait.offline": "サーバーにつながりません。もう一度ためしています…",
	"gone.title": "時間です",
	"gone.note": "遊んでくれてありがとう！このアヒルはもう休みました。",
	"gone.crash": "アヒルのシミュレーターが急に止まりました。ごめんなさい。",
	"gone.start": "アヒルが目をさましませんでした。ごめんなさい。",
	"gone.idle": "しばらくはなれていたので、アヒルは先に休みました。",
	"gone.place": "長くはなれていたので、順番がなくなりました。",
	"gone.left": "あなたがはなれたので、アヒルは休みました。",
	"gone.again": "もう一羽",
};

const es: Catalog = {
	"page.title": "Simulador del patito · prueba pública",
	"page.brand": "Simulador del patito",
	"page.lang": "Idioma",
	"land.eyebrow": "Prueba pública",
	"land.h1a": "Cría un patito",
	"land.h1b": "durante {minutes} minutos",
	"land.lead": "Cada visitante recibe un patito nuevo y al azar para jugar en su propio simulador durante {minutes} minutos. Al terminar, puedes descargarlo como archivo .duck y quedártelo.",
	"land.how1": "Pasa la comprobación de que eres una persona y pulsa el botón.",
	"land.how2": "Si hay un sitio libre, empiezas enseguida; si no, esperas en la fila y esta página te dice tu puesto.",
	"land.how3": "El patito tarda un poco en despertar, uno o dos minutos.",
	"land.stats": "Ahora juegan {live} patitos y esperan {queued} personas (como máximo {max} a la vez).",
	"land.start": "Quiero un patito",
	"land.verify": "Primero completa la comprobación de arriba y luego pulsa el botón.",
	"land.nc.title": "Sin fines comerciales",
	"land.nc.text": "Esta prueba es gratis: sin anuncios ni pagos. El modelo 3D del patito solo se puede usar sin fines comerciales.",
	"land.attr": "El modelo 3D del patito es el Microduck de Pollen Robotics, con licencia CC BY-NC-SA 4.0. duckmind es un proyecto no oficial, sin relación con Pollen Robotics.",
	"land.license": "Licencia",
	"land.privacy": "Usamos una sola cookie para reconocer tu prueba; tu dirección de red se guarda solo como hash y por poco tiempo, para limitar cuántas veces puede empezar.",
	"err.turnstile": "La comprobación no pasó. Inténtalo otra vez.",
	"err.already": "Ya tienes un patito jugando o esperando en la fila.",
	"err.rate": "Acabas de empezar varios patitos. Vuelve un poco más tarde.",
	"err.busy": "Ahora mismo hay demasiada gente esperando. Vuelve un poco más tarde.",
	"err.start": "El patito no despertó (un problema del servidor). Inténtalo otra vez.",
	"queue.title": "En la fila",
	"queue.place": "Eres el número {place}",
	"queue.note": "Cada patito juega {minutes} minutos. Cuando sea tu turno, esta página abrirá tu patito sola. No cierres esta página: si te vas un minuto, pierdes tu puesto.",
	"queue.leave": "Salir de la fila",
	"start.title": "Tu patito se está despertando",
	"start.note": "Estamos preparando un patito nuevo para ti; suele tardar uno o dos minutos. Esta página lo abrirá sola cuando esté listo.",
	"wait.offline": "No se puede conectar con el servidor, reintentando…",
	"gone.title": "Se acabó el tiempo",
	"gone.note": "¡Gracias por jugar! Este patito ya se fue a descansar.",
	"gone.crash": "El simulador del patito se detuvo de repente. Lo sentimos.",
	"gone.start": "El patito no despertó. Lo sentimos.",
	"gone.idle": "Estuviste fuera un rato, así que el patito se fue a descansar.",
	"gone.place": "Estuviste fuera demasiado tiempo y perdiste tu puesto en la fila.",
	"gone.left": "Te fuiste y el patito se fue a descansar.",
	"gone.again": "Otro patito",
};

const zhHans: Catalog = {
	"page.title": "小鸭模拟器・公开试玩",
	"page.brand": "小鸭模拟器",
	"page.lang": "语言",
	"land.eyebrow": "公开试玩",
	"land.h1a": "养一只小鸭",
	"land.h1b": "{minutes} 分钟",
	"land.lead": "每位访客会拿到一只全新的随机小鸭，在自己的模拟器里玩 {minutes} 分钟。时间到了，可以把它下载成 .duck 文件带走。",
	"land.how1": "完成真人验证，按下按钮。",
	"land.how2": "有空位就马上开始；没有的话就排队，这页会告诉你排第几。",
	"land.how3": "小鸭醒来要一点时间，大约一两分钟。",
	"land.stats": "现在有 {live} 只小鸭在玩、{queued} 人在排队（最多同时 {max} 只）。",
	"land.start": "给我一只小鸭",
	"land.verify": "先完成上面的验证，再按按钮。",
	"land.nc.title": "非商业用途",
	"land.nc.text": "这个试玩免费、没有广告、不收费。小鸭的 3D 模型只能用在非商业用途。",
	"land.attr": "小鸭的 3D 模型来自 Pollen Robotics 的 Microduck，授权是 CC BY-NC-SA 4.0。duckmind 是非官方项目，与 Pollen Robotics 无关。",
	"land.license": "授权条款",
	"land.privacy": "我们只用一个 cookie 认出你的试玩；你的网络地址只以哈希的形式暂存，用来限制次数。",
	"err.turnstile": "验证没有通过，请再试一次。",
	"err.already": "你已经有一只小鸭在玩或在排队了。",
	"err.rate": "你刚刚已经开过好几只小鸭了，请晚一点再来。",
	"err.busy": "现在排队的人太多了，请晚一点再来。",
	"err.start": "小鸭没有醒来（服务器出了问题），请再试一次。",
	"queue.title": "排队中",
	"queue.place": "你排第 {place} 位",
	"queue.note": "每只小鸭玩 {minutes} 分钟。轮到你时，这页会自己打开小鸭。请不要关掉这页：离开一分钟就会失去位置。",
	"queue.leave": "离开队伍",
	"start.title": "小鸭正在醒来",
	"start.note": "正在帮你准备一只新的小鸭，通常要一两分钟。准备好了，这页会自己打开。",
	"wait.offline": "连不上服务器，正在重试…",
	"gone.title": "时间到了",
	"gone.note": "谢谢你来玩！这只小鸭已经去休息了。",
	"gone.crash": "小鸭的模拟器意外停了，对不起。",
	"gone.start": "小鸭没有醒来，对不起。",
	"gone.idle": "你离开了一阵子，小鸭先去休息了。",
	"gone.place": "你离开太久，失去了排队的位置。",
	"gone.left": "你离开了，小鸭去休息了。",
	"gone.again": "再来一只",
};

const ko: Catalog = {
	"page.title": "오리 시뮬레이터 · 공개 체험",
	"page.brand": "오리 시뮬레이터",
	"page.lang": "언어",
	"land.eyebrow": "공개 체험",
	"land.h1a": "오리 키우기",
	"land.h1b": "{minutes}분 동안",
	"land.lead": "방문한 사람마다 새로운 무작위 오리 한 마리를 받아, 자기만의 시뮬레이터에서 {minutes}분 동안 놀 수 있어요. 시간이 끝나면 .duck 파일로 내려받아 가져갈 수 있어요.",
	"land.how1": "사람 확인을 마치고 버튼을 누르세요.",
	"land.how2": "자리가 있으면 바로 시작하고, 없으면 줄을 서요. 이 페이지에 몇 번째인지 나와요.",
	"land.how3": "오리가 깨어나는 데 1~2분쯤 걸려요.",
	"land.stats": "지금 {live}마리가 놀고 있고 {queued}명이 기다리고 있어요(동시에 최대 {max}마리).",
	"land.start": "오리 받기",
	"land.verify": "먼저 위의 확인을 마친 다음 버튼을 누르세요.",
	"land.nc.title": "비상업적 이용",
	"land.nc.text": "이 체험은 무료이고 광고도 요금도 없어요. 오리의 3D 모델은 비상업적 용도로만 쓸 수 있어요.",
	"land.attr": "오리의 3D 모델은 Pollen Robotics의 Microduck이고, 라이선스는 CC BY-NC-SA 4.0이에요. duckmind는 비공식 프로젝트이며 Pollen Robotics와 관계가 없어요.",
	"land.license": "라이선스",
	"land.privacy": "체험을 알아보려고 쿠키 하나만 써요. 네트워크 주소는 횟수를 제한하려고 해시 형태로 잠시만 보관해요.",
	"err.turnstile": "확인을 통과하지 못했어요. 다시 해 주세요.",
	"err.already": "이미 놀고 있거나 줄을 서 있는 오리가 있어요.",
	"err.rate": "방금 오리를 여러 번 시작했어요. 조금 뒤에 다시 와 주세요.",
	"err.busy": "지금 기다리는 사람이 너무 많아요. 조금 뒤에 다시 와 주세요.",
	"err.start": "오리가 깨어나지 않았어요(서버 문제). 다시 해 주세요.",
	"queue.title": "줄 서는 중",
	"queue.place": "{place}번째예요",
	"queue.note": "오리 한 마리당 {minutes}분 동안 놀아요. 차례가 오면 이 페이지가 저절로 오리를 열어요. 이 페이지를 닫지 마세요. 1분 동안 떠나 있으면 순서를 잃어요.",
	"queue.leave": "줄에서 나가기",
	"start.title": "오리가 깨어나고 있어요",
	"start.note": "새 오리를 준비하고 있어요. 보통 1~2분 걸려요. 준비되면 이 페이지가 저절로 열어요.",
	"wait.offline": "서버에 연결할 수 없어요. 다시 시도하는 중…",
	"gone.title": "시간이 끝났어요",
	"gone.note": "놀아 줘서 고마워요! 이 오리는 이제 쉬러 갔어요.",
	"gone.crash": "오리의 시뮬레이터가 갑자기 멈췄어요. 미안해요.",
	"gone.start": "오리가 깨어나지 않았어요. 미안해요.",
	"gone.idle": "한동안 자리를 비워서 오리가 먼저 쉬러 갔어요.",
	"gone.place": "너무 오래 자리를 비워서 줄의 순서를 잃었어요.",
	"gone.left": "떠나셔서 오리가 쉬러 갔어요.",
	"gone.again": "한 마리 더",
};

export const CATALOGS: Record<Lang, Catalog> = {
	"zh-Hant": zhHant,
	en,
	ja,
	es,
	"zh-Hans": zhHans,
	ko,
};

/** The message in `lang` (繁體中文 when missing) with `{name}` values filled in. */
export function t(lang: Lang, key: string, values: Record<string, string | number> = {}): string {
	const text = CATALOGS[lang][key] ?? zhHant[key] ?? key;
	return text.replace(/\{(\w+)\}/g, (whole, name: string) =>
		Object.hasOwn(values, name) ? String(values[name]) : whole,
	);
}

function asLang(value: string | null | undefined): Lang | null {
	if (!value) return null;
	const v = value.trim().toLowerCase();
	for (const l of LANGS) if (l.toLowerCase() === v) return l;
	if (v === "zh-tw" || v === "zh-hk" || v === "zh-mo") return "zh-Hant";
	if (v === "zh-cn" || v === "zh-sg") return "zh-Hans";
	return null;
}

/** ?lang=, then the visitor's saved choice (cookie dm_lang), then Accept-Language, then 繁體中文. */
export function pickLang(query: string | null, saved: string | null, acceptLanguage: string | null): Lang {
	const chosen = asLang(query) ?? asLang(saved);
	if (chosen) return chosen;
	for (const part of (acceptLanguage ?? "").split(",")) {
		const tag = part.split(";")[0]?.trim() ?? "";
		const exact = asLang(tag);
		if (exact) return exact;
		const base = tag.split("-")[0]?.toLowerCase();
		if (base === "zh") return tag.toLowerCase().includes("hans") ? "zh-Hans" : "zh-Hant";
		const match = LANGS.find((l) => l === base);
		if (match) return match;
	}
	return "zh-Hant";
}
