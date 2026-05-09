# goktugGPT — Roadmap & To-Do

Bu dosya projenin geleceğini planlar. Her madde teknik detaylarıyla açıklanmıştır.
Şu an yapılmıyor — ileriye yönelik geliştirme listesi.

---

## 1. Veriyi 1M+ Satıra Çıkar

**Hedef:** Şu an 175K satır var. 1M+ satır ile model GPT-2 small kalitesine gerçekten ulaşır.

**Kaynaklar:**
- **OpenWebText** — Reddit'te en az 3 karma almış linklerin içerikleri. ~40GB, ~8B token. Hugging Face'te mevcut.
- **Common Crawl** — İnternetin ham kopyası. Petabyte ölçeğinde. `cc_net` veya `OSCAR` filtrelenmiş versiyonu kullanılır.
- **BookCorpus** — 11,000 kitap, ~1B token. GPT-1/2'nin eğitim verisinin bir parçasıydı.
- **Wikipedia dump** — Zaten `download_data.py`'da var ama genişletilebilir.

**Yapılacak:**
- `download_conversations.py`'a OpenWebText downloader ekle (Hugging Face datasets API)
- Raw text'i `<user>...<assistant>...` formatına çevirmek yerine plain text olarak da eğitilebilir (pretraining ayrı, finetuning ayrı)
- 2 aşamalı pipeline: önce raw text ile pretrain, sonra konuşma verisiyle finetune

**Komut hedefi:**
```bash
python data/download_big.py --openwebtext --wikipedia-full
python train.py --config large --data data/pretrain.txt --epochs 5   # pretrain
python train.py --config large --data data/train_chat.txt --resume --epochs 20  # finetune
```

---

## 2. Reasoning Model — Gerçek Thinking

**Hedef:** `<think>` bloğunu şablon doldurmaktan çıkarıp gerçek adım adım akıl yürütmeye dönüştür.

**Teknik yaklaşım:**
Tam RLHF çok karmaşık. Daha uygulanabilir alternatif: **STaR (Self-Taught Reasoner)**

1. Modele matematik/mantık problemleri ver
2. Model `<think>` bloğunda adımları yaz, sonra cevap ver
3. Doğru cevaba ulaştığında o `<think>` bloğunu eğitim verisine ekle
4. Yanlış cevapta bloğu at
5. Tekrarla — model zamanla kendi doğru reasoning zincirlerini üretiyor

**Veri kaynakları:**
- GSM8K — 8,500 matematik word problem + çözüm adımları
- MATH dataset — zor matematik problemleri
- CommonsenseQA — günlük mantık soruları
- ARC — bilim soruları

**Train.txt formatı şöyle genişler:**
```
<user> If John has 3 apples and gives 1 to Mary, how many does he have?
<assistant> <think>
John starts with 3 apples. He gives 1 to Mary. 3 - 1 = 2.
</think> John has 2 apples. <eos>
```

**Komut hedefi:**
```bash
python data/download_reasoning.py   # GSM8K + MATH + ARC indir
python train.py --config large --data data/train_reasoning.txt --resume
```

---

## 3. Modern GUI — ChatGPT/Claude Tarzı Arayüz

**Hedef:** Mevcut `gui.py`'ı tamamen yenile. Profesyonel, kişisel hissettiren bir arayüz.

**Teknoloji seçenekleri:**
- **Option A: Gradio** — Hızlı, Python native, kurulumu kolay. Ama sınırlı özelleştirme.
- **Option B: Streamlit** — Daha fazla kontrol, component sistemi iyi.
- **Option C: React frontend + FastAPI backend** — Tam kontrol, gerçek web app. En çok iş ama en iyi sonuç.
- **Tavsiye: Streamlit** kısa vadede, ileride React'a geç.

**Özellikler:**
- Sol sidebar: chat history listesi (isimler otomatik üretilir)
- Sağ panel: aktif konuşma
- Dark/light mode toggle
- Mesaj baloncukları (user sağda, assistant solda)
- Thinking bloğu açılır/kapanır (collapsible)
- Dosya yükleme butonu (RAG için)
- Sohbet içi arama (Ctrl+F)
- Model bilgisi göstergesi (parametre sayısı, config)
- "New Chat" butonu
- Konuşmayı export et (JSON/TXT)
- Yazarken "goktugGPT is thinking..." animasyonu
- Kullanıcı adı ayarı ("Merhaba, Göktug!")
- Mesaj kopyalama butonu

**Dosya yapısı:**
```
src/
  gui/
    app.py          ← mevcut (eski)
    streamlit_app.py  ← yeni modern GUI
    components/
      sidebar.py
      chat_window.py
      message_bubble.py
    styles/
      theme.css
```

---

## 4. Conversation Memory — Bağlam Hafızası

**Hedef:** Session içinde söylenen her şeyi hatırla. "Ben Göktug" → "Sen Göktug'sun."

**Teknik durum:**
Bu özellik aslında transformer'da zaten var — context window'a geçmiş mesajları koymak yeterli.
Sorun: mevcut `chat.py` her mesajı bağımsız işliyor olabilir.

**Yapılacak:**
- `chat.py`'da conversation history buffer tut
- Her yeni mesajda önceki konuşmayı context'e ekle:
```
<user> Benim adım Göktug <assistant> <think>...</think> Merhaba Göktug! <eos>
<user> Ben kimim? <assistant> <think> Kullanıcı daha önce adının Göktug olduğunu söyledi. </think> Sen Göktug'sun! <eos>
```
- Max context length aşılınca en eski mesajları at (sliding window)
- Önemli bilgileri (isim, tercihler) ayrı bir "user profile" dict'te sakla
- Session'lar arası kalıcı hafıza için SQLite veya JSON dosyası

**Uzun vadede:**
- Vektör veritabanı (ChromaDB, FAISS) ile semantic memory
- "Bunu hatırla" komutuyla kalıcı kayıt

---

## 5. RAG — Dosya Yükleme ve Sorgulama

**Hedef:** Kullanıcı PDF/TXT yüklesin, model o belgeden soru yanıtlasın.

**Teknik akış:**
1. Kullanıcı dosya yükler (PDF, TXT, DOCX)
2. Dosya chunk'lara bölünür (her chunk ~200 token)
3. Her chunk embedding vektörüne dönüştürülür (SentenceTransformers ile)
4. Kullanıcı soru sorar
5. Sorunun embedding'i hesaplanır
6. En yakın chunk'lar bulunur (cosine similarity)
7. Bu chunk'lar context'e eklenir, model cevap üretir

**Kütüphaneler:**
- `sentence-transformers` — embedding üretmek için (all-MiniLM-L6-v2 modeli, küçük ve hızlı)
- `faiss-cpu` — vektör araması
- `pypdf2` veya `pdfminer` — PDF okuma
- `python-docx` — Word dosyası okuma

**Komut hedefi:**
```python
# GUI'de dosya yükleme butonu
# Arkada:
from src.rag import RAGSystem
rag = RAGSystem()
rag.load_document("belge.pdf")
answer = rag.query("Bu belgede ne anlatılıyor?", model)
```

---

## 6. Web Search Tool

**Hedef:** Model gerçek zamanlı bilgiye erişsin. "Bugün hava nasıl?" sorusuna cevap verebilsin.

**Teknik akış:**
1. Model `<search>sorgu buraya</search>` token üretir
2. Sistem bu token'ı yakalar
3. DuckDuckGo veya SerpAPI ile arama yapılır
4. İlk 3-5 sonucun özeti context'e eklenir
5. Model devam eder ve cevap üretir

**Tool use eğitimi:**
Modelin `<search>` token üretmeyi öğrenmesi için training data'ya örnekler eklemek lazım:
```
<user> Tesla'nın bugünkü hisse fiyatı ne?
<assistant> <think> Bu güncel bilgi, search lazım. </think>
<search> Tesla stock price today </search>
<search_result> TSLA: $245.30 (+2.1%) </search_result>
Tesla'nın bugünkü hisse fiyatı 245.30 dolar, %2.1 artışla. <eos>
```

**Araçlar:**
- `duckduckgo-search` Python kütüphanesi (API key gerektirmez)
- Alternatif: SerpAPI (günlük 100 ücretsiz sorgu)

---

## 7. Devasa README

**Hedef:** Projeyi anlatan, teknik derinliği olan, görsel açıdan şık bir README.

**İçerik planı:**

### Bölümler:
1. **Proje Tanımı** — goktugGPT nedir, ne yapar, neden yapıldı
2. **Mimari** — Transformer decoder-only, katman sayıları, attention mekanizması diyagramı
3. **Sistem Nasıl Çalışır** — Token → Embedding → Attention → FFN → Softmax → Sampling
4. **Thinking Mekanizması** — `<think>` bloğu ne işe yarıyor
5. **Eğitim Süreci** — BPE tokenizer, loss function, optimizer, warmup schedule
6. **Veri Seti** — Kaynaklar, format, boyut
7. **Config Karşılaştırması** — tiny/default/medium/large tablosu
8. **Kurulum** — pip install, gereksinimler
9. **Kullanım** — train.py, chat.py, gui.py komutları
10. **Colab Eğitimi** — Adım adım Google Colab kılavuzu
11. **Gelecek Planlar** — Bu roadmap'in özeti
12. **Teknik Kaynaklar** — Attention Is All You Need, GPT-2 paper, vs.

---

## Öncelik Sırası (Önerim)

| # | Özellik | Etki | Zorluk | Önce mi? |
|---|---------|------|--------|----------|
| 1 | Conversation Memory | Yüksek | Kolay | ✅ İlk yap |
| 2 | Modern GUI (Streamlit) | Yüksek | Orta | ✅ İkinci |
| 3 | README | Orta | Kolay | ✅ Üçüncü |
| 4 | Veri 1M+ satır | Çok Yüksek | Orta | Eğitim sonrası |
| 5 | Web Search | Yüksek | Orta | Sonra |
| 6 | RAG | Yüksek | Zor | Sonra |
| 7 | Reasoning (STaR) | Yüksek | Çok Zor | En son |

---

*Bu dosya yaşayan bir belge — her geliştirme adımında güncellenir.*
