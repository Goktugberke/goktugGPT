#!/usr/bin/env python3
"""
generate_data.py — Generate large-scale training data for goktugGPT

Sources:
  1. Stanford Alpaca dataset (52K instruction-following examples)
  2. Algorithmically generated math problems
  3. Turkish knowledge & conversation
  4. Multi-turn conversations (name memory, topic continuation)
  5. goktugGPT identity examples

Usage:
    python generate_data.py                          # Full generation
    python generate_data.py --skip-alpaca            # Skip Alpaca download
    python generate_data.py --output data/custom.txt

After generating:
    1. Delete old tokenizer:   del checkpoints\\tokenizer.json
    2. Delete old checkpoints: del checkpoints\\*.pt
    3. Retrain:                python train.py --config large
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
from typing import List

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
DEFAULT_OUTPUT = "data/train.txt"
BACKUP_PATH = "data/train_original_backup.txt"
ALPACA_CACHE = "data/alpaca_data.json"

random.seed(42)

# ══════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ══════════════════════════════════════════════════════════════════════

def _oneline(text: str) -> str:
    """Collapse newlines and excess whitespace into single spaces."""
    return " ".join(text.split())


def fmt(user: str, think: str, answer: str) -> str:
    u = _oneline(user)
    t = _oneline(think)
    a = _oneline(answer)
    if t:
        return f"<user> {u} <assistant> <think> {t} </think> {a} <eos>"
    return f"<user> {u} <assistant> {a} <eos>"


def fmt_multi(turns: List[dict]) -> str:
    parts = []
    for i, turn in enumerate(turns):
        u = _oneline(turn["user"])
        t = _oneline(turn.get("think", ""))
        a = _oneline(turn["answer"])
        if t:
            part = f"<user> {u} <assistant> <think> {t} </think> {a}"
        else:
            part = f"<user> {u} <assistant> {a}"
        if i == len(turns) - 1:
            part += " <eos>"
        parts.append(part)
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════
#  1. ALPACA
# ══════════════════════════════════════════════════════════════════════

def download_alpaca(cache: str = ALPACA_CACHE) -> list:
    if os.path.exists(cache):
        print(f"  Cached: {cache}")
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)
    print("  Downloading Alpaca dataset...")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    urllib.request.urlretrieve(ALPACA_URL, cache)
    with open(cache, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Downloaded {len(data):,} examples.")
    return data


def _make_thinking(category: str, instruction: str, output: str) -> str:
    """Generate thinking text that references the actual question content."""
    inst_short = instruction[:80].strip()
    # Extract a key noun/topic from the instruction for specificity
    topic_words = [w for w in instruction.split() if len(w) > 4 and w[0].isalpha()]
    topic = topic_words[0] if topic_words else ""

    if category == "math":
        return random.choice([
            f"Let me solve this math problem step by step.",
            f"I need to calculate this. Let me work through it carefully.",
            f"Math problem about {topic.lower()}. Let me compute." if topic else "Math problem. Let me compute.",
        ])
    if category == "writing":
        return random.choice([
            f"I need to write about {topic.lower()}. Let me think about structure and tone." if topic else "Writing task. Let me plan the structure.",
            f"The user wants me to write something. Let me think about what would be most effective.",
            f"Writing request. I should focus on clarity and quality.",
            f"Let me plan this writing carefully before starting.",
        ])
    if category == "translation":
        return random.choice([
            "Translation request. I need to preserve the original meaning while making it natural.",
            "Let me translate this accurately and naturally.",
        ])
    if category == "summary":
        return random.choice([
            "I need to identify the key points and condense them.",
            "Summary request. Let me extract the essential information.",
            "Let me focus on the main ideas and present them concisely.",
        ])
    if category == "explanation":
        return random.choice([
            f"The user is asking about {topic.lower()}. Let me explain this clearly." if topic else "Let me explain this clearly.",
            f"I should break this down into understandable parts.",
            f"Explanation needed. Let me think about the best way to present this.",
            f"Let me think about {topic.lower()} and give a clear explanation." if topic else "Let me give a clear explanation.",
        ])
    if category == "listing":
        return random.choice([
            f"I need to come up with relevant items for this list.",
            f"Let me think of good examples to include.",
            f"List request about {topic.lower()}. Let me recall the key items." if topic else "List request. Let me think of key items.",
        ])
    if category == "coding":
        return random.choice([
            f"Programming question. Let me think about the logic and best approach.",
            f"I need to think about the right algorithm and implementation.",
            f"Code question about {topic.lower()}. Let me plan the solution." if topic else "Code question. Let me plan the solution.",
            f"Let me think about how to implement this correctly.",
        ])
    if category == "comparison":
        return random.choice([
            "I need to identify the key similarities and differences.",
            "Let me compare these objectively, looking at strengths and weaknesses.",
            "Comparison request. Let me analyze both sides.",
        ])
    if category == "reasoning":
        return random.choice([
            f"Let me think through this logically, step by step.",
            f"I need to reason about {topic.lower()} carefully." if topic else "I need to reason through this carefully.",
            f"Good question. Let me think about the key factors.",
            f"Let me consider the different aspects of this question.",
        ])
    # general
    return random.choice([
        f"Let me think about this question about {topic.lower()} carefully." if topic else "Let me think about this carefully.",
        f"I should consider what would be most helpful here.",
        f"Let me think about the best way to answer this.",
        f"The user is asking about {topic.lower()}. Let me provide a helpful response." if topic else "Let me provide a helpful response.",
    ])


def _classify(instruction: str) -> str:
    low = instruction.lower()
    if any(w in low for w in ["calculate", "compute", "sum ", "multiply", "divide"]) and any(c.isdigit() for c in instruction):
        return "math"
    if any(w in low for w in ["write", "compose", "draft", "create a", "generate a"]):
        return "writing"
    if "translate" in low:
        return "translation"
    if any(w in low for w in ["summarize", "summary", "sum up"]):
        return "summary"
    if any(w in low for w in ["explain", "describe", "what is", "what are", "define"]):
        return "explanation"
    if any(w in low for w in ["list", "name ", "enumerate", "give me"]):
        return "listing"
    if any(w in low for w in ["code", "function", "program", "python", "javascript", "html", "sql", "implement"]):
        return "coding"
    if any(w in low for w in ["compare", "difference", "contrast", "versus"]):
        return "comparison"
    if any(w in low for w in ["why", "how does", "how do", "how can"]):
        return "reasoning"
    return "general"


def convert_alpaca(data: list, variations: int = 1) -> List[str]:
    """Convert Alpaca data. variations>1 creates multiple thinking variants per example."""
    prefixes = ["", "Please ", "Can you ", "Could you ", "I need you to "]
    examples = []
    for item in data:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output or len(output) > 1000 or len(output) < 5:
            continue
        category = _classify(instruction)
        for v in range(variations):
            if v == 0:
                user_msg = f"{instruction} {input_text}" if input_text else instruction
            else:
                prefix = random.choice(prefixes)
                inst = instruction[0].lower() + instruction[1:] if prefix else instruction
                user_msg = f"{prefix}{inst} {input_text}" if input_text else f"{prefix}{inst}"
            think = _make_thinking(category, instruction, output)
            examples.append(fmt(user_msg, think, output))
    return examples


# ══════════════════════════════════════════════════════════════════════
#  2. MATH
# ══════════════════════════════════════════════════════════════════════

def generate_math(n: int = 150000) -> List[str]:
    examples = []

    # Arithmetic
    ops = [("+", "plus", lambda a, b: a + b),
           ("-", "minus", lambda a, b: a - b),
           ("*", "times", lambda a, b: a * b)]
    for _ in range(n // 5):
        a, b = random.randint(1, 500), random.randint(1, 500)
        sym, word, fn = random.choice(ops)
        r = fn(a, b)
        q = random.choice([f"What is {a} {word} {b}?", f"Calculate {a} {sym} {b}", f"{a} {sym} {b} = ?"])
        examples.append(fmt(q, f"{a} {sym} {b} = {r}.", f"{a} {word} {b} equals {r}."))

    # Division
    for _ in range(n // 10):
        b = random.randint(2, 50)
        r = random.randint(1, 100)
        a = b * r
        q = random.choice([f"What is {a} divided by {b}?", f"Calculate {a} / {b}"])
        examples.append(fmt(q, f"{a} / {b} = {r}.", f"{a} divided by {b} equals {r}."))

    # Percentages
    for _ in range(n // 10):
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        base = random.choice([50, 100, 200, 300, 400, 500, 1000])
        r = int(pct * base / 100)
        q = random.choice([f"What is {pct}% of {base}?", f"Calculate {pct} percent of {base}."])
        examples.append(fmt(q, f"{pct}/100 * {base} = {r}.", f"{pct}% of {base} is {r}."))

    # Powers
    for _ in range(n // 10):
        base = random.randint(2, 20)
        exp = random.choice([2, 3])
        r = base ** exp
        word = "squared" if exp == 2 else "cubed"
        q = random.choice([f"What is {base} {word}?", f"Calculate {base}^{exp}."])
        examples.append(fmt(q, f"{base}^{exp} = {r}.", f"{base} {word} is {r}."))

    # Prime check
    for _ in range(n // 10):
        num = random.randint(2, 200)
        is_prime = all(num % i != 0 for i in range(2, int(num**0.5) + 1)) and num > 1
        q = f"Is {num} a prime number?"
        if is_prime:
            examples.append(fmt(q, f"{num} is only divisible by 1 and itself.", f"Yes, {num} is a prime number."))
        else:
            f2 = next(i for i in range(2, num) if num % i == 0)
            examples.append(fmt(q, f"{num} = {f2} * {num//f2}. Not prime.", f"No, {num} is not prime. It is divisible by {f2}."))

    # Word problems
    names = ["Alice", "Bob", "Charlie", "Emma", "Sarah", "John", "Goktug", "Mehmet", "Ayse"]
    items = ["apples", "books", "cookies", "pencils", "marbles", "stickers", "candies"]
    for _ in range(n // 10):
        name, item = random.choice(names), random.choice(items)
        a, b = random.randint(3, 50), random.randint(1, 30)
        if random.random() < 0.5:
            total = a + b
            q = f"{name} has {a} {item}. They buy {b} more. How many {item} does {name} have now?"
            examples.append(fmt(q, f"{a} + {b} = {total}.", f"{name} now has {total} {item}."))
        else:
            b = min(b, a)
            total = a - b
            q = f"{name} has {a} {item}. They give away {b}. How many {item} does {name} have now?"
            examples.append(fmt(q, f"{a} - {b} = {total}.", f"{name} now has {total} {item}."))

    # Sequences
    for _ in range(n // 10):
        start, step = random.randint(1, 20), random.randint(1, 10)
        seq = [start + step * i for i in range(6)]
        nxt = start + step * 6
        q = f"What comes next: {', '.join(str(x) for x in seq)}, ...?"
        examples.append(fmt(q, f"Pattern: +{step}. {seq[-1]} + {step} = {nxt}.", f"The next number is {nxt}."))

    # Turkish math
    for _ in range(n // 10):
        a, b = random.randint(1, 200), random.randint(1, 200)
        if random.random() < 0.5:
            r = a + b
            q = random.choice([f"{a} artı {b} kaç eder?", f"{a} + {b} kaçtır?"])
            examples.append(fmt(q, f"{a} + {b} = {r}.", f"{a} artı {b} eşittir {r}."))
        else:
            if b > a: a, b = b, a
            r = a - b
            q = random.choice([f"{a} eksi {b} kaçtır?", f"{a} - {b} kaç eder?"])
            examples.append(fmt(q, f"{a} - {b} = {r}.", f"{a} eksi {b} eşittir {r}."))

    random.shuffle(examples)
    return examples[:n]


# ══════════════════════════════════════════════════════════════════════
#  3. TURKISH DATA
# ══════════════════════════════════════════════════════════════════════

TURKEY_CITIES = [
    ("İstanbul", "Marmara Bölgesi", "Türkiye'nin en kalabalık şehridir. Nüfusu yaklaşık 16 milyondur. Boğaziçi ile Avrupa ve Asya'yı birleştirir."),
    ("Ankara", "İç Anadolu Bölgesi", "Türkiye'nin başkentidir. 1923'te başkent ilan edilmiştir."),
    ("İzmir", "Ege Bölgesi", "Türkiye'nin üçüncü büyük şehridir. Ege kıyısında yer alır."),
    ("Antalya", "Akdeniz Bölgesi", "Türkiye'nin turizm başkentidir. Güneş, kum ve deniz ile ünlüdür."),
    ("Bursa", "Marmara Bölgesi", "Osmanlı İmparatorluğu'nun ilk başkentidir. Uludağ ile ünlüdür."),
    ("Trabzon", "Karadeniz Bölgesi", "Sümela Manastırı ile ünlü tarihi bir şehirdir."),
    ("Konya", "İç Anadolu Bölgesi", "Mevlana Celaleddin Rumi'nin yaşadığı şehirdir."),
    ("Gaziantep", "Güneydoğu Anadolu Bölgesi", "UNESCO gastronomi şehridir. Yemek kültürü ile dünyaca ünlüdür."),
    ("Diyarbakır", "Güneydoğu Anadolu Bölgesi", "Tarihi surları ile ünlüdür. Dicle Nehri kıyısındadır."),
    ("Eskişehir", "İç Anadolu Bölgesi", "Öğrenci şehri olarak bilinir. Porsuk Çayı şehrin ortasından geçer."),
    ("Edirne", "Marmara Bölgesi", "Selimiye Camii ile ünlüdür. Osmanlı'nın ikinci başkentidir."),
    ("Kayseri", "İç Anadolu Bölgesi", "Erciyes Dağı ve mantısı ile meşhurdur."),
    ("Samsun", "Karadeniz Bölgesi", "Atatürk'ün 19 Mayıs 1919'da çıkarak Kurtuluş Savaşı'nı başlattığı şehirdir."),
    ("Adana", "Akdeniz Bölgesi", "Adana kebabı ile meşhurdur. Seyhan Nehri kıyısındadır."),
    ("Mardin", "Güneydoğu Anadolu Bölgesi", "Taş evleri ve Mezopotamya manzarası ile ünlüdür."),
    ("Van", "Doğu Anadolu Bölgesi", "Van Gölü Türkiye'nin en büyük gölüdür. Van kahvaltısı ile ünlüdür."),
    ("Çanakkale", "Marmara Bölgesi", "Çanakkale Savaşı'nın yaşandığı yerdir. Truva antik kenti burada bulunur."),
    ("Erzurum", "Doğu Anadolu Bölgesi", "Doğu'nun en büyük şehirlerinden biridir. Cağ kebabı ile meşhurdur."),
    ("Hatay", "Akdeniz Bölgesi", "Medeniyetler beşiği olarak bilinir. Künefe tatlısı ile ünlüdür."),
    ("Muğla", "Ege Bölgesi", "Bodrum ve Marmaris gibi tatil beldeleriyle bilinir."),
]

TURKEY_FACTS = [
    ("Osmanlı İmparatorluğu ne zaman kuruldu?", "1299'da Osman Gazi tarafından kurulmuştur.", "Osmanlı kuruluşu. 1299, Osman Gazi."),
    ("İstanbul ne zaman fethedildi?", "29 Mayıs 1453'te Fatih Sultan Mehmet tarafından fethedilmiştir.", "İstanbul'un fethi. 1453, Fatih Sultan Mehmet."),
    ("Türkiye Cumhuriyeti ne zaman kuruldu?", "29 Ekim 1923'te kurulmuştur. Mustafa Kemal Atatürk ilk cumhurbaşkanı olmuştur.", "Cumhuriyet kuruluşu. 29 Ekim 1923."),
    ("Mustafa Kemal Atatürk kimdir?", "Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır. Kurtuluş Savaşı'nı yönetmiş ve modern Türkiye'nin temellerini atmıştır.", "Atatürk. Cumhuriyetin kurucusu."),
    ("Kurtuluş Savaşı ne zaman oldu?", "1919-1923 yılları arasında yapılmıştır. Atatürk önderliğinde bağımsızlık kazanılmıştır.", "Kurtuluş Savaşı. 1919-1923."),
    ("Çanakkale Savaşı ne zaman oldu?", "1915 yılında olmuştur. Osmanlı büyük bir zafer kazanmıştır. Çanakkale geçilmez!", "Çanakkale Savaşı. 1915."),
    ("Harf devrimi ne zaman yapıldı?", "1 Kasım 1928'de yapılmıştır. Arap harfleri yerine Latin harfleri kabul edilmiştir.", "Harf devrimi. 1928."),
    ("Kadınlara seçme hakkı ne zaman verildi?", "1934 yılında verilmiştir. Birçok Avrupa ülkesinden önce tanınmıştır.", "Kadın hakları. 1934."),
    ("Türkiye hangi kıtada?", "Hem Avrupa hem Asya kıtasında yer alır. Anadolu Asya'da, Trakya Avrupa'dadır.", "Türkiye'nin konumu. İki kıta."),
    ("Türkiye'nin başkenti neresidir?", "Türkiye'nin başkenti Ankara'dır.", "Başkent. Ankara."),
    ("Türkiye'nin en yüksek dağı hangisidir?", "Ağrı Dağı'dır. Yüksekliği 5137 metredir.", "En yüksek dağ. Ağrı Dağı, 5137m."),
    ("Türkiye'nin en büyük gölü hangisidir?", "Van Gölü'dür.", "En büyük göl. Van Gölü."),
    ("Türkiye'nin en uzun nehri hangisidir?", "Kızılırmak'tır. 1355 kilometre uzunluğundadır.", "En uzun nehir. Kızılırmak."),
    ("Türkiye'de kaç bölge vardır?", "7 coğrafi bölge: Marmara, Ege, Akdeniz, Karadeniz, İç Anadolu, Doğu Anadolu, Güneydoğu Anadolu.", "7 bölge."),
    ("Suyun kimyasal formülü nedir?", "H2O. İki hidrojen ve bir oksijen atomundan oluşur.", "Su = H2O."),
    ("DNA nedir?", "Deoksiribonükleik asit. Canlıların genetik bilgisini taşıyan çift sarmal yapılı moleküldür.", "DNA. Genetik bilgi taşıyıcı."),
    ("Fotosentez nedir?", "Bitkilerin güneş ışığını kullanarak CO2 ve suyu glikoz ve oksijene dönüştürdüğü süreçtir.", "Fotosentez. Bitkilerin besin üretmesi."),
    ("Yerçekimi nedir?", "Kütleli cisimleri birbirine çeken temel bir kuvvettir. Bizi yere bağlar ve gezegenleri yörüngede tutar.", "Yerçekimi. Kütleçekim kuvveti."),
    ("Atom nedir?", "Maddenin en küçük yapı taşıdır. Proton, nötron ve elektronlardan oluşur.", "Atom. Maddenin yapı taşı."),
    ("Deprem nedir?", "Yer kabuğundaki fay hatları boyunca biriken enerjinin aniden serbest kalmasıyla oluşan sarsıntıdır.", "Deprem. Fay hatları, enerji."),
    ("Yapay zeka nedir?", "Bilgisayarların insan zekasını taklit eden görevleri yerine getirmesini sağlayan bilgisayar bilimi alanıdır.", "Yapay zeka tanımı."),
    ("Mevlana kimdir?", "13. yüzyılda yaşamış ünlü bir sufi düşünür ve şairdir. Konya'da yaşamış, Mesnevi'yi yazmıştır.", "Mevlana. 13. yüzyıl, Konya."),
    ("Nasreddin Hoca kimdir?", "13. yüzyılda yaşadığı düşünülen ünlü bir mizah ustasıdır. Fıkraları zekice mesajlar içerir.", "Nasreddin Hoca. Mizah ustası."),
    ("Türk kahvesi nasıl yapılır?", "Cezve ile yapılır. İnce öğütülmüş kahve ve su kısık ateşte kaynatılır. Köpüğü ile fincanara dökülür.", "Türk kahvesi. Cezve, kısık ateş."),
    ("Türk çayı nasıl yapılır?", "Çift demlik ile yapılır. Alt kısımda su kaynar, üstte çay demlenir. İnce belli bardakta servis edilir.", "Türk çayı. Çift demlik."),
    ("Bayramlar nelerdir?", "Dini: Ramazan Bayramı, Kurban Bayramı. Milli: 23 Nisan, 19 Mayıs, 30 Ağustos, 29 Ekim.", "Bayramlar. Dini ve milli."),
]

TURKISH_GREETINGS = [
    ("Merhaba", "Selamlama.", "Merhaba! Size nasıl yardımcı olabilirim?"),
    ("Selam", "Kısa selamlama.", "Selam! Bugün nasıl yardım edebilirim?"),
    ("Günaydın", "Sabah selamı.", "Günaydın! Güzel bir gün olsun."),
    ("İyi akşamlar", "Akşam selamı.", "İyi akşamlar! Nasıl yardımcı olabilirim?"),
    ("Nasılsın", "Hal hatır.", "İyiyim, teşekkür ederim! Size nasıl yardımcı olabilirim?"),
    ("Naber", "Samimi selam.", "İyidir! Sen nasılsın?"),
    ("Teşekkürler", "Teşekkür.", "Rica ederim! Başka sorunuz varsa çekinmeyin."),
    ("Görüşürüz", "Veda.", "Görüşürüz! İyi günler dilerim."),
    ("Hoşça kal", "Veda.", "Hoşça kalın! Tekrar beklerim."),
    ("Sen kimsin?", "Kimlik sorusu.", "Ben goktugGPT, sıfırdan inşa edilmiş yerel bir dil modeliyim."),
    ("Ne yapabilirsin?", "Yetenek sorusu.", "Sorulara cevap verebilir, sohbet edebilir, bilgi sağlayabilirim. Türkçe ve İngilizce konuşurum."),
    ("Türkçe konuşabiliyor musun?", "Dil sorusu.", "Evet, Türkçe konuşabiliyorum! Hem Türkçe hem İngilizce yardımcı olabilirim."),
    ("Bana bir fıkra anlat", "Fıkra isteği.", "Temel balığa gitmiş. Oltayı suya atmış. Dursun sormuş: 'Ne yapıyorsun?' Temel: 'Balık tutuyorum.' Dursun: 'Neden suya atıyorsun?' Temel: 'Kuru balık mı tutayım!'"),
]


def generate_turkish() -> List[str]:
    examples = []

    # City questions
    for city, region, desc in TURKEY_CITIES:
        examples.append(fmt(f"{city} hakkında bilgi ver.", f"{city}. {region}.", f"{city} {region}'nde yer alır. {desc}"))
        examples.append(fmt(f"{city} nerededir?", f"{city} konumu.", f"{city} {region}'nde yer almaktadır."))
        examples.append(fmt(f"{city} hangi bölgededir?", f"{city} bölgesi.", f"{city} {region}'nde yer alır."))

    # Facts (history, science, culture)
    for q, a, think in TURKEY_FACTS:
        examples.append(fmt(q, think, a))

    # Greetings & daily
    for q, think, a in TURKISH_GREETINGS:
        examples.append(fmt(q, think, a))

    # Colors
    colors = {"kırmızı": "red", "mavi": "blue", "yeşil": "green", "sarı": "yellow",
              "siyah": "black", "beyaz": "white", "turuncu": "orange", "mor": "purple"}
    for tr, en in colors.items():
        examples.append(fmt(f"'{tr}' İngilizce ne demek?", f"{tr} = {en}.", f"'{tr}' İngilizce '{en}' demektir."))
        examples.append(fmt(f"'{en}' Türkçe ne demek?", f"{en} = {tr}.", f"'{en}' Türkçe '{tr}' demektir."))

    # Days & months
    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    months = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
    examples.append(fmt("Haftanın günleri nelerdir?", "Günler.", "Haftanın günleri: " + ", ".join(days) + "."))
    examples.append(fmt("Yılın ayları nelerdir?", "Aylar.", "Yılın ayları: " + ", ".join(months) + "."))

    # Proverbs
    proverbs = [
        ("Damlaya damlaya göl olur ne demek?", "Küçük birikimler büyük sonuçlar doğurur. Sabır ve süreklilik önemlidir.", "Atasözü. Birikim."),
        ("Ağaç yaşken eğilir ne demek?", "Eğitim küçük yaşta başlamalıdır. İnsanlar gençken daha kolay şekillenir.", "Atasözü. Erken eğitim."),
        ("Bir elin nesi var iki elin sesi var ne demek?", "Birlik ve beraberlik güçtür. İş birliği başarı getirir.", "Atasözü. Birlik."),
        ("Sakla samanı gelir zamanı ne demek?", "Tutumlu olmak ve biriktirmek önemlidir. Saklanan şey ileride işe yarar.", "Atasözü. Tutumluluk."),
    ]
    for q, a, think in proverbs:
        examples.append(fmt(q, think, a))

    return examples


# ══════════════════════════════════════════════════════════════════════
#  4. MULTI-TURN CONVERSATIONS
# ══════════════════════════════════════════════════════════════════════

NAMES_EN = ["Alice", "Bob", "Charlie", "David", "Emma", "Sarah", "John", "Maria",
            "James", "Lisa", "Michael", "Rachel", "Daniel", "Sophie"]
NAMES_TR = ["Goktug", "Mehmet", "Ayse", "Fatma", "Ali", "Zeynep", "Emre",
            "Elif", "Burak", "Selin", "Deniz", "Cem", "Hakan", "Merve"]
ALL_NAMES = NAMES_EN + NAMES_TR

COUNTRIES = [
    ("Turkey", "Ankara"), ("United States", "Washington D.C."), ("France", "Paris"),
    ("Germany", "Berlin"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"),
    ("Italy", "Rome"), ("Spain", "Madrid"), ("South Korea", "Seoul"),
    ("China", "Beijing"), ("India", "New Delhi"), ("United Kingdom", "London"),
]


def generate_multiturn() -> List[str]:
    examples = []

    # --- Name memory ---
    for name in ALL_NAMES:
        examples.append(fmt_multi([
            {"user": f"Hi, my name is {name}",
             "think": f"User introduced themselves as {name}.",
             "answer": f"Nice to meet you, {name}! How can I help you today?"},
            {"user": "What is my name?",
             "think": f"The user asked for their name. They said {name} earlier.",
             "answer": f"Your name is {name}!"},
        ]))

    # --- Name + country memory ---
    for name in ALL_NAMES:
        country, capital = random.choice(COUNTRIES)
        examples.append(fmt_multi([
            {"user": f"My name is {name} and I am from {country}",
             "think": f"User: {name}, from {country}.",
             "answer": f"Hello {name}! Nice to meet someone from {country}. How can I help?"},
            {"user": "Where am I from?",
             "think": f"They said they are from {country}.",
             "answer": f"You are from {country}!"},
            {"user": "And what is my name?",
             "think": f"Their name is {name}.",
             "answer": f"Your name is {name}!"},
        ]))

    # --- Name + topic ---
    for name in ALL_NAMES:
        country, capital = random.choice(COUNTRIES)
        examples.append(fmt_multi([
            {"user": f"Hello, I'm {name}",
             "think": f"Greeting. Name is {name}.",
             "answer": f"Hello {name}! What can I do for you?"},
            {"user": f"What is the capital of {country}?",
             "think": f"Capital of {country}: {capital}.",
             "answer": f"The capital of {country} is {capital}."},
            {"user": "Thanks! Do you remember my name?",
             "think": f"Name recall. They are {name}.",
             "answer": f"Of course! Your name is {name}. Happy to help!"},
        ]))

    # --- Preference memory ---
    colors = ["blue", "red", "green", "purple", "yellow"]
    foods = ["pizza", "sushi", "pasta", "kebab", "tacos", "ramen"]
    for name in random.sample(ALL_NAMES, 14):
        color, food = random.choice(colors), random.choice(foods)
        examples.append(fmt_multi([
            {"user": f"My name is {name} and my favorite color is {color}",
             "think": f"Name: {name}, fav color: {color}.",
             "answer": f"Nice to meet you, {name}! {color.capitalize()} is a great color!"},
            {"user": f"I also love {food}",
             "think": f"Also likes {food}. {name}, color={color}, food={food}.",
             "answer": f"Great taste! {food.capitalize()} is delicious."},
            {"user": "What do you know about me?",
             "think": f"Recall: {name}, color={color}, food={food}.",
             "answer": f"Your name is {name}, your favorite color is {color}, and you love {food}!"},
        ]))

    # --- Topic continuation ---
    topic_flows = [
        [{"user": "What is Python?", "think": "Python programming language.", "answer": "Python is a high-level programming language known for its readable syntax. It is widely used in AI, web development, and data science."},
         {"user": "Is it good for beginners?", "think": "Python beginner-friendliness.", "answer": "Yes! Python is one of the best languages for beginners. Its syntax is clean and reads almost like English."},
         {"user": "What can I build with it?", "think": "Python use cases.", "answer": "With Python you can build websites, AI models, data pipelines, games, automation scripts, and much more."}],

        [{"user": "How many planets are there?", "think": "Solar system. 8 planets.", "answer": "There are 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},
         {"user": "Which is the largest?", "think": "Largest planet: Jupiter.", "answer": "Jupiter is the largest planet. Over 1,300 Earths could fit inside it."},
         {"user": "And the smallest?", "think": "Smallest planet: Mercury.", "answer": "Mercury is the smallest planet, only slightly larger than Earth's Moon."}],

        [{"user": "What is artificial intelligence?", "think": "AI definition.", "answer": "Artificial intelligence is the field of computer science focused on building machines that can perform tasks requiring human intelligence."},
         {"user": "What is deep learning?", "think": "Deep learning: subset of ML, neural networks.", "answer": "Deep learning uses neural networks with many layers. Each layer learns increasingly abstract features. It powers modern AI breakthroughs."},
         {"user": "Is that what you use?", "think": "My architecture: transformer, deep learning.", "answer": "Yes! I am built using a transformer architecture, which is a type of deep learning model using self-attention."}],

        [{"user": "What is water made of?", "think": "Water: H2O.", "answer": "Water is made of two hydrogen atoms and one oxygen atom: H2O."},
         {"user": "At what temperature does it boil?", "think": "Boiling: 100C / 212F.", "answer": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard pressure."},
         {"user": "And freeze?", "think": "Freezing: 0C / 32F.", "answer": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit)."}],

        [{"user": "When was World War 2?", "think": "WW2: 1939-1945.", "answer": "World War 2 lasted from 1939 to 1945."},
         {"user": "Who won?", "think": "Allied Powers won WW2.", "answer": "The Allied Powers won: the US, UK, Soviet Union, France, and China."},
         {"user": "What happened after?", "think": "Post-WW2: Cold War, UN, decolonization.", "answer": "The United Nations was formed, the Cold War began between the US and USSR, and many colonies gained independence."}],

        [{"user": "How do I make a website?", "think": "Web dev basics: HTML, CSS, JS.", "answer": "You need HTML for structure, CSS for styling, and JavaScript for interactivity."},
         {"user": "What is HTML?", "think": "HTML: HyperText Markup Language.", "answer": "HTML is the standard language for web pages. It uses tags like <h1>, <p>, and <a> to define content structure."},
         {"user": "And CSS?", "think": "CSS: styling language.", "answer": "CSS controls how HTML looks: colors, fonts, layouts, spacing. It makes websites visually appealing."}],
    ]
    for flow in topic_flows:
        examples.append(fmt_multi(flow))
        examples.append(fmt_multi(flow[:2]))

    # --- Turkish multi-turn ---
    for name in NAMES_TR:
        examples.append(fmt_multi([
            {"user": f"Merhaba, benim adım {name}",
             "think": f"Kullanıcı: {name}.",
             "answer": f"Merhaba {name}! Nasıl yardımcı olabilirim?"},
            {"user": "Benim adım neydi?",
             "think": f"Adı {name}.",
             "answer": f"Senin adın {name}!"},
        ]))

        city, region, desc = random.choice(TURKEY_CITIES)
        examples.append(fmt_multi([
            {"user": f"Selam, ben {name}",
             "think": f"Selamlama. Ad: {name}.",
             "answer": f"Selam {name}! Nasıl yardımcı olabilirim?"},
            {"user": f"{city} hakkında bilgi verir misin?",
             "think": f"{city}. {region}.",
             "answer": f"{city} {region}'nde yer alır. {desc}"},
            {"user": "Teşekkürler! Adımı hatırlıyor musun?",
             "think": f"Ad hatırlama: {name}.",
             "answer": f"Tabii ki! Senin adın {name}. Başka sorun varsa çekinme!"},
        ]))

    # --- Mixed language ---
    for name in NAMES_TR[:7]:
        examples.append(fmt_multi([
            {"user": f"Hi, I'm {name} from Turkey",
             "think": f"User: {name}, from Turkey.",
             "answer": f"Hello {name}! Nice to meet you."},
            {"user": "Türkiye hakkında ne biliyorsun?",
             "think": "Switched to Turkish. Turkey info.",
             "answer": "Türkiye, Avrupa ve Asya kıtalarında yer alan bir ülkedir. Başkenti Ankara, en büyük şehri İstanbul'dur."},
            {"user": "Thanks! What was my name?",
             "think": f"Back to English. Name: {name}.",
             "answer": f"Your name is {name}!"},
        ]))

    return examples


# ══════════════════════════════════════════════════════════════════════
#  5. GENERAL KNOWLEDGE (templated, scalable)
# ══════════════════════════════════════════════════════════════════════

WORLD_COUNTRIES = [
    ("Afghanistan", "Kabul", "Asia", "Pashto and Dari"),
    ("Albania", "Tirana", "Europe", "Albanian"),
    ("Algeria", "Algiers", "Africa", "Arabic"),
    ("Argentina", "Buenos Aires", "South America", "Spanish"),
    ("Australia", "Canberra", "Oceania", "English"),
    ("Austria", "Vienna", "Europe", "German"),
    ("Bangladesh", "Dhaka", "Asia", "Bengali"),
    ("Belgium", "Brussels", "Europe", "Dutch, French, German"),
    ("Brazil", "Brasilia", "South America", "Portuguese"),
    ("Canada", "Ottawa", "North America", "English and French"),
    ("Chile", "Santiago", "South America", "Spanish"),
    ("China", "Beijing", "Asia", "Mandarin Chinese"),
    ("Colombia", "Bogota", "South America", "Spanish"),
    ("Croatia", "Zagreb", "Europe", "Croatian"),
    ("Cuba", "Havana", "North America", "Spanish"),
    ("Czech Republic", "Prague", "Europe", "Czech"),
    ("Denmark", "Copenhagen", "Europe", "Danish"),
    ("Egypt", "Cairo", "Africa", "Arabic"),
    ("Ethiopia", "Addis Ababa", "Africa", "Amharic"),
    ("Finland", "Helsinki", "Europe", "Finnish and Swedish"),
    ("France", "Paris", "Europe", "French"),
    ("Germany", "Berlin", "Europe", "German"),
    ("Greece", "Athens", "Europe", "Greek"),
    ("Hungary", "Budapest", "Europe", "Hungarian"),
    ("Iceland", "Reykjavik", "Europe", "Icelandic"),
    ("India", "New Delhi", "Asia", "Hindi and English"),
    ("Indonesia", "Jakarta", "Asia", "Indonesian"),
    ("Iran", "Tehran", "Asia", "Persian"),
    ("Iraq", "Baghdad", "Asia", "Arabic and Kurdish"),
    ("Ireland", "Dublin", "Europe", "Irish and English"),
    ("Israel", "Jerusalem", "Asia", "Hebrew and Arabic"),
    ("Italy", "Rome", "Europe", "Italian"),
    ("Japan", "Tokyo", "Asia", "Japanese"),
    ("Kenya", "Nairobi", "Africa", "Swahili and English"),
    ("Malaysia", "Kuala Lumpur", "Asia", "Malay"),
    ("Mexico", "Mexico City", "North America", "Spanish"),
    ("Morocco", "Rabat", "Africa", "Arabic"),
    ("Netherlands", "Amsterdam", "Europe", "Dutch"),
    ("New Zealand", "Wellington", "Oceania", "English and Maori"),
    ("Nigeria", "Abuja", "Africa", "English"),
    ("North Korea", "Pyongyang", "Asia", "Korean"),
    ("Norway", "Oslo", "Europe", "Norwegian"),
    ("Pakistan", "Islamabad", "Asia", "Urdu and English"),
    ("Peru", "Lima", "South America", "Spanish"),
    ("Philippines", "Manila", "Asia", "Filipino and English"),
    ("Poland", "Warsaw", "Europe", "Polish"),
    ("Portugal", "Lisbon", "Europe", "Portuguese"),
    ("Romania", "Bucharest", "Europe", "Romanian"),
    ("Russia", "Moscow", "Europe/Asia", "Russian"),
    ("Saudi Arabia", "Riyadh", "Asia", "Arabic"),
    ("Singapore", "Singapore", "Asia", "English, Malay, Mandarin, Tamil"),
    ("South Africa", "Pretoria", "Africa", "11 official languages including English, Zulu, Afrikaans"),
    ("South Korea", "Seoul", "Asia", "Korean"),
    ("Spain", "Madrid", "Europe", "Spanish"),
    ("Sweden", "Stockholm", "Europe", "Swedish"),
    ("Switzerland", "Bern", "Europe", "German, French, Italian, Romansh"),
    ("Thailand", "Bangkok", "Asia", "Thai"),
    ("Turkey", "Ankara", "Europe/Asia", "Turkish"),
    ("Ukraine", "Kyiv", "Europe", "Ukrainian"),
    ("United Arab Emirates", "Abu Dhabi", "Asia", "Arabic"),
    ("United Kingdom", "London", "Europe", "English"),
    ("United States", "Washington D.C.", "North America", "English"),
    ("Venezuela", "Caracas", "South America", "Spanish"),
    ("Vietnam", "Hanoi", "Asia", "Vietnamese"),
]

ANIMALS = [
    ("dog", "Dogs are domesticated mammals and one of the most popular pets. They are known for loyalty and intelligence. There are over 300 breeds."),
    ("cat", "Cats are small domesticated carnivores. They are popular pets known for independence and agility. They can sleep up to 16 hours a day."),
    ("elephant", "Elephants are the largest land animals. They are highly intelligent with excellent memory. African elephants have larger ears than Asian elephants."),
    ("lion", "Lions are large cats known as the king of the jungle. They live in groups called prides. Males have distinctive manes."),
    ("eagle", "Eagles are large birds of prey with excellent eyesight. They can spot prey from over 2 miles away. The bald eagle is the national bird of the United States."),
    ("dolphin", "Dolphins are highly intelligent marine mammals. They communicate using clicks and whistles. They are known for playful behavior."),
    ("whale", "Whales are the largest animals on Earth. Blue whales can reach 30 meters in length. They are marine mammals that breathe air."),
    ("penguin", "Penguins are flightless birds that live mostly in the Southern Hemisphere. Emperor penguins can survive temperatures below -60 degrees Celsius."),
    ("tiger", "Tigers are the largest wild cats. They have distinctive orange and black stripes. They are solitary hunters found mainly in Asia."),
    ("shark", "Sharks are fish that have been around for over 400 million years. Great white sharks can detect a drop of blood in 25 gallons of water."),
    ("giraffe", "Giraffes are the tallest land animals, reaching up to 5.5 meters tall. Their long necks help them reach leaves in tall trees."),
    ("snake", "Snakes are legless reptiles found on every continent except Antarctica. Some species are venomous while others are constrictors."),
    ("bear", "Bears are large mammals found across the world. There are 8 species including polar bears, grizzly bears, and pandas."),
    ("wolf", "Wolves are social carnivores that live in packs. They communicate through howling and are ancestors of domestic dogs."),
    ("horse", "Horses are large domesticated mammals used for riding and work for thousands of years. They can sleep both standing up and lying down."),
    ("octopus", "Octopuses are intelligent marine animals with eight arms, three hearts, and blue blood. They can change color and squeeze through tiny spaces."),
    ("bee", "Bees are flying insects crucial for pollination. A single bee colony can pollinate 300 million flowers each day. Honey bees produce honey."),
    ("crocodile", "Crocodiles are large reptiles that have existed since the time of dinosaurs. They have the strongest bite force of any animal."),
    ("monkey", "Monkeys are primates found in tropical regions. They are intelligent and social, living in groups. There are over 260 species."),
    ("turtle", "Turtles are reptiles with protective shells. Sea turtles can live over 100 years. They have existed for over 200 million years."),
]

ELEMENTS = [
    ("Hydrogen", "H", 1, "the lightest element and most abundant in the universe"),
    ("Helium", "He", 2, "a noble gas used in balloons and cooling systems"),
    ("Carbon", "C", 6, "the basis of all organic life"),
    ("Nitrogen", "N", 7, "makes up 78% of Earth's atmosphere"),
    ("Oxygen", "O", 8, "essential for breathing, makes up 21% of Earth's atmosphere"),
    ("Sodium", "Na", 11, "a soft metal; sodium chloride is table salt"),
    ("Iron", "Fe", 26, "used in steel production; essential for blood hemoglobin"),
    ("Gold", "Au", 79, "a precious metal valued for jewelry and electronics"),
    ("Silver", "Ag", 47, "a precious metal with the highest electrical conductivity"),
    ("Copper", "Cu", 29, "widely used in electrical wiring due to good conductivity"),
    ("Aluminum", "Al", 13, "lightweight metal used in cans, foil, and aircraft"),
    ("Silicon", "Si", 14, "key element in computer chips and semiconductors"),
    ("Uranium", "U", 92, "used as fuel in nuclear power plants"),
    ("Platinum", "Pt", 78, "a rare precious metal used in catalytic converters"),
    ("Lead", "Pb", 82, "a dense metal historically used in pipes and paint"),
]

INVENTIONS = [
    ("telephone", "Alexander Graham Bell", "1876"),
    ("light bulb", "Thomas Edison", "1879"),
    ("airplane", "Wright Brothers", "1903"),
    ("internet", "ARPANET / Tim Berners-Lee (WWW)", "1969 / 1991"),
    ("printing press", "Johannes Gutenberg", "1440"),
    ("steam engine", "James Watt (improved)", "1769"),
    ("television", "Philo Farnsworth", "1927"),
    ("penicillin", "Alexander Fleming", "1928"),
    ("computer", "Charles Babbage (concept) / Alan Turing (theory)", "1837 / 1936"),
    ("electricity", "Benjamin Franklin (studied) / Thomas Edison (practical use)", "1752 / 1882"),
    ("telescope", "Hans Lippershey / Galileo Galilei (improved)", "1608"),
    ("vaccine", "Edward Jenner", "1796"),
    ("dynamite", "Alfred Nobel", "1867"),
    ("radio", "Guglielmo Marconi", "1895"),
    ("automobile", "Karl Benz", "1886"),
]


def generate_knowledge() -> List[str]:
    """Generate diverse general knowledge Q&A from data dictionaries."""
    examples = []

    # --- Countries ---
    for country, capital, continent, language in WORLD_COUNTRIES:
        # Capital
        q_templates = [
            f"What is the capital of {country}?",
            f"Tell me the capital of {country}.",
            f"Capital of {country}?",
        ]
        for q in q_templates:
            examples.append(fmt(q, f"Capital of {country}: {capital}.",
                                f"The capital of {country} is {capital}."))

        # Continent
        examples.append(fmt(f"What continent is {country} in?",
                            f"{country} location.",
                            f"{country} is located in {continent}."))

        # Language
        examples.append(fmt(f"What language is spoken in {country}?",
                            f"{country} language.",
                            f"The main language spoken in {country} is {language}."))

        # Combined
        examples.append(fmt(f"Tell me about {country}.",
                            f"{country}: capital {capital}, {continent}.",
                            f"{country} is a country in {continent}. Its capital is {capital} and the main language is {language}."))

    # --- Animals ---
    for animal, desc in ANIMALS:
        q_templates = [
            f"Tell me about {animal}s.",
            f"What do you know about {animal}s?",
            f"What is a {animal}?",
            f"Describe a {animal}.",
        ]
        for q in q_templates:
            examples.append(fmt(q, f"{animal.capitalize()} facts.", desc))

    # --- Elements ---
    for name, symbol, number, desc in ELEMENTS:
        examples.append(fmt(f"What is {name}?",
                            f"{name}: element {number}, symbol {symbol}.",
                            f"{name} (symbol: {symbol}, atomic number: {number}) is a chemical element. It is {desc}."))
        examples.append(fmt(f"What is the chemical symbol for {name}?",
                            f"{name} symbol: {symbol}.",
                            f"The chemical symbol for {name} is {symbol}."))
        examples.append(fmt(f"What is element number {number}?",
                            f"Element {number}: {name}.",
                            f"Element number {number} is {name} ({symbol}). It is {desc}."))

    # --- Inventions ---
    for invention, inventor, year in INVENTIONS:
        examples.append(fmt(f"Who invented the {invention}?",
                            f"{invention}: {inventor}, {year}.",
                            f"The {invention} was invented by {inventor} in {year}."))
        examples.append(fmt(f"When was the {invention} invented?",
                            f"{invention}: {year}.",
                            f"The {invention} was invented in {year} by {inventor}."))

    # --- Unit conversions ---
    conversions = [
        ("How many centimeters are in a meter?", "100 cm = 1 m.", "There are 100 centimeters in a meter."),
        ("How many meters are in a kilometer?", "1000 m = 1 km.", "There are 1,000 meters in a kilometer."),
        ("How many grams are in a kilogram?", "1000 g = 1 kg.", "There are 1,000 grams in a kilogram."),
        ("How many milliliters are in a liter?", "1000 mL = 1 L.", "There are 1,000 milliliters in a liter."),
        ("How many seconds are in a minute?", "60 seconds.", "There are 60 seconds in a minute."),
        ("How many minutes are in an hour?", "60 minutes.", "There are 60 minutes in an hour."),
        ("How many hours are in a day?", "24 hours.", "There are 24 hours in a day."),
        ("How many days are in a year?", "365 days (366 in leap year).", "There are 365 days in a year, or 366 in a leap year."),
        ("How many bytes are in a kilobyte?", "1024 bytes.", "There are 1,024 bytes in a kilobyte."),
        ("How many bits are in a byte?", "8 bits.", "There are 8 bits in a byte."),
    ]
    for q, think, a in conversions:
        examples.append(fmt(q, think, a))

    # --- Programming concepts (templated) ---
    prog_concepts = [
        ("What is a variable?", "A variable is a named storage location in memory that holds a value which can change during program execution."),
        ("What is a function?", "A function is a reusable block of code that performs a specific task. It takes inputs, processes them, and can return outputs."),
        ("What is a loop?", "A loop is a control structure that repeats a block of code. For loops iterate a fixed number of times, while loops run while a condition is true."),
        ("What is an array?", "An array is a data structure that stores a collection of elements in contiguous memory. Elements are accessed by index."),
        ("What is a string?", "A string is a sequence of characters used to represent text. In most languages, strings are enclosed in quotes."),
        ("What is a boolean?", "A boolean is a data type with only two values: true or false. It is used for logical operations and conditions."),
        ("What is an API?", "An API (Application Programming Interface) is a set of rules that allows different software systems to communicate with each other."),
        ("What is a database?", "A database is an organized collection of structured data stored electronically. It allows efficient storage, retrieval, and manipulation of data."),
        ("What is version control?", "Version control is a system that tracks changes to files over time. Git is the most popular version control system."),
        ("What is debugging?", "Debugging is the process of finding and fixing errors (bugs) in code. It involves identifying the cause of unexpected behavior."),
        ("What is an IDE?", "An IDE (Integrated Development Environment) is a software application for writing code. It typically includes a code editor, debugger, and build tools."),
        ("What is object-oriented programming?", "OOP is a paradigm that organizes code into objects containing data (attributes) and behavior (methods). Key concepts: classes, inheritance, encapsulation, polymorphism."),
        ("What is recursion?", "Recursion is when a function calls itself to solve a problem by breaking it into smaller instances. Every recursive function needs a base case to stop."),
        ("What is a linked list?", "A linked list is a data structure where each element (node) contains data and a pointer to the next node. Unlike arrays, elements are not stored contiguously."),
        ("What is a hash table?", "A hash table is a data structure that maps keys to values using a hash function. It provides O(1) average time for lookups, insertions, and deletions."),
        ("What is Big O notation?", "Big O notation describes the worst-case time or space complexity of an algorithm as input size grows. For example, O(n) means linear time."),
        ("What is a binary tree?", "A binary tree is a data structure where each node has at most two children. Binary search trees keep data sorted for efficient searching."),
        ("What is machine learning?", "Machine learning is a type of AI where computers learn patterns from data instead of being explicitly programmed. It improves with more data."),
        ("What is a neural network?", "A neural network is a computing system inspired by the brain, with layers of connected nodes. It learns by adjusting connection weights during training."),
        ("What is an operating system?", "An operating system is software that manages hardware and provides services for applications. Examples: Windows, macOS, Linux."),
    ]
    for q, a in prog_concepts:
        think = "Programming concept. Let me explain clearly."
        examples.append(fmt(q, think, a))
        examples.append(fmt(f"Explain: {q.lower().replace('what is ', '').rstrip('?')}",
                            think, a))

    return examples


# ══════════════════════════════════════════════════════════════════════
#  6. IDENTITY
# ══════════════════════════════════════════════════════════════════════

def generate_identity() -> List[str]:
    examples = []
    qa = [
        ("Who are you?", "I am goktugGPT, a language model built from scratch using transformer architecture. I run locally on your machine.", "I am goktugGPT."),
        ("What is goktugGPT?", "goktugGPT is a locally trained language model built from scratch. It runs on your device without internet.", "goktugGPT explanation."),
        ("Who created you?", "I was created as a from-scratch AI project using Python and PyTorch.", "Creator question."),
        ("Are you like ChatGPT?", "Similar concept — we both use transformers. But I am much smaller and run locally without internet.", "ChatGPT comparison."),
        ("Can you access the internet?", "No. I run entirely locally. All my knowledge comes from training data.", "Internet: no."),
        ("How do you work?", "I use a transformer decoder with multi-head self-attention. I process tokens and generate responses one token at a time.", "Architecture question."),
        ("What languages do you speak?", "I can communicate in English and Turkish.", "Language capabilities."),
        ("Are you conscious?", "No. I am a mathematical model that predicts the next word based on training patterns. I have no awareness.", "Consciousness: no."),
    ]
    for q, a, think in qa:
        examples.append(fmt(q, think, a))

    tr_qa = [
        ("Sen kimsin?", "Ben goktugGPT, sıfırdan inşa edilmiş yerel bir dil modeliyim.", "Kimlik. goktugGPT."),
        ("goktugGPT nedir?", "Python ve PyTorch ile sıfırdan oluşturulmuş, transformer tabanlı yerel bir dil modelidir.", "goktugGPT nedir."),
        ("Nasıl çalışıyorsun?", "Transformer decoder mimarisi kullanıyorum. Metni tokenlara ayırıp dikkat mekanizması ile yanıt üretiyorum.", "Çalışma prensibi."),
        ("İnternete bağlı mısın?", "Hayır, tamamen yerel çalışıyorum. Bilgilerim eğitim verilerimden geliyor.", "İnternet: hayır."),
    ]
    for q, a, think in tr_qa:
        examples.append(fmt(q, think, a))

    return examples


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate training data for goktugGPT")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-alpaca", action="store_true")
    parser.add_argument("--math-count", type=int, default=150000)
    parser.add_argument("--alpaca-variations", type=int, default=3,
                        help="Thinking-text variations per Alpaca example (1-5)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    all_examples = []

    # 1. Alpaca (augmented with multiple thinking variations)
    if not args.skip_alpaca:
        print(f"[1/6] Alpaca dataset (x{args.alpaca_variations} variations)...")
        try:
            raw = download_alpaca()
            alpaca = convert_alpaca(raw, variations=args.alpaca_variations)
            print(f"  {len(alpaca):,} examples")
            all_examples.extend(alpaca)
        except Exception as e:
            print(f"  Warning: {e}")
            print("  Continuing without Alpaca...")
    else:
        print("[1/6] Alpaca... SKIPPED")

    # 2. Math
    print(f"[2/6] Math problems ({args.math_count:,})...")
    math_ex = generate_math(args.math_count)
    print(f"  {len(math_ex):,} examples")
    all_examples.extend(math_ex)

    # 3. General knowledge
    print("[3/6] General knowledge...")
    knowledge = generate_knowledge()
    print(f"  {len(knowledge):,} examples")
    all_examples.extend(knowledge)

    # 4. Turkish
    print("[4/6] Turkish data...")
    tr = generate_turkish()
    print(f"  {len(tr):,} examples")
    all_examples.extend(tr)

    # 5. Multi-turn
    print("[5/6] Multi-turn conversations...")
    mt = generate_multiturn()
    print(f"  {len(mt):,} examples")
    all_examples.extend(mt)

    # 6. Identity
    print("[6/6] Identity examples...")
    ident = generate_identity()
    print(f"  {len(ident):,} examples")
    all_examples.extend(ident)

    # 7. Backup old data (do NOT re-ingest — generated data replaces it entirely)
    if os.path.exists(args.output):
        print("[7/7] Backing up old data...")
        shutil.copy2(args.output, BACKUP_PATH)
        print(f"  Backed up to {BACKUP_PATH}")
    else:
        print("[7/7] No old data to back up.")

    # Shuffle and write
    random.shuffle(all_examples)

    print(f"\n{'='*50}")
    print(f"  TOTAL: {len(all_examples):,} examples")
    print(f"{'='*50}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# goktugGPT Training Data\n")
        f.write(f"# Auto-generated: {len(all_examples):,} examples\n")
        f.write("# Format: <user> question <assistant> <think> reasoning </think> answer <eos>\n\n")
        for ex in all_examples:
            f.write(ex + "\n")

    print(f"\nWritten to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Delete old tokenizer + checkpoints:")
    print(f"       del checkpoints\\tokenizer.json")
    print(f"       del checkpoints\\*.pt")
    print(f"  2. Retrain:")
    print(f"       python train.py --config large")


if __name__ == "__main__":
    main()
